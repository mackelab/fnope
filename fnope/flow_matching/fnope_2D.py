import torch
import torch.nn as nn
from torch import Tensor
from fnope.nets.finite_embeddings import FiniteXEmbedding
from fnope.nets.time_embeddings import TimestepEmbedding
from fnope.nets.positional_embeddings import PositionEmbedding
from fnope.nets.fno_blocks import FNO2DBlock_DSE,FNO2DBlock_DSE_FixedContext
from fnope.nets.spectral_transforms import SpectralTransform,FFT2D, VFT2D
from fnope.nets.padding2D import get_padding
from fnope.nets.finite_nets import VectorFieldMLP
from fnope.nets.standardizing_net import FiniteStandardizing, IdentityStandardizing, FilterStandardizing2d
from fnope.flow_matching.base_distributions import WhiteNoise, FrequencyThresholdedGaussianProcess2d
from typing import Union,Optional, Tuple
from warnings import warn
import math

import zuko
from zuko.distributions import DiagNormal, NormalizingFlow, Joint
from zuko.transforms import FreeFormJacobianTransform


class Velocity_Unified2D(nn.Module):
    def __init__(self,
                modes: int,
                in_channels: int,
                ctx_in_channels: int,
                conv_channels: int,
                x_finite_dim: Optional[int] = None,
                ctx_embedding_channels: int =16,
                time_embedding_channels: int = 8,
                position_embedding_channels: Optional[int] = 8,
                num_layers: int = 5,
                act = nn.GELU(),
                padding = {"type":"zero","pad_length":10},
                always_equispaced = False,
                
                **kwargs):
        
        super().__init__()

        self.modes = modes
        self.in_channels = in_channels
        self.conv_channels = conv_channels # if conv channels comes as a list, the number of entries must be same as num_layer --> add error message
        self.ctx_in_channels = ctx_in_channels
        self.ctx_embedding_channels = ctx_embedding_channels
        self.time_embedding_channels = time_embedding_channels
        self.position_embedding_channels = position_embedding_channels
        self.num_layers = num_layers
        self.always_equispaced = always_equispaced

        self.time_embedding_net = TimestepEmbedding(modes, hidden_dim=64, output_dim = time_embedding_channels)
        if self.position_embedding_channels is not None:
            self.position_embedding_net = PositionEmbedding(domain_dim=2,hidden_dim=64, output_dim=position_embedding_channels, act=act)
            self.ctx_position_embedding_net = PositionEmbedding(domain_dim=2,hidden_dim=64, output_dim=position_embedding_channels, act=act)
        self.padding = get_padding(**padding)

        conv_module = nn.Conv2d if self.always_equispaced else nn.Conv1d
        self.x_input_layer = conv_module(self.in_channels,self.conv_channels,1)
        self.ctx_input_layer = conv_module(self.ctx_in_channels,self.ctx_embedding_channels,1)
        #Make the ctx processing layer a bit more flexible by adding one FNO block with itself
        self.ctx_self_block = FNO2DBlock_DSE(in_channels=self.ctx_embedding_channels,
                                             out_channels=self.ctx_embedding_channels,
                                             modes=self.modes,
                                             equispaced=self.always_equispaced,
                                             act=act,
                                             time_embedding_channels=self.time_embedding_channels,
                                             pos_embedding_channels=self.position_embedding_channels,
                                             )


        # Implement option for variable conv channel number here if conv channels comes as a list
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(FNO2DBlock_DSE_FixedContext(in_channels=self.conv_channels,
                                                           ctx_channels=self.ctx_embedding_channels, 
                                                           out_channels = self.conv_channels,
                                                           modes = self.modes, 
                                                           equispaced = self.always_equispaced,
                                                           act=act,
                                                           time_embedding_channels = self.time_embedding_channels,
                                                           pos_embedding_channels = self.position_embedding_channels,
                                                           conditional_info_channels = self.ctx_embedding_channels,
                                                           )
                                               )
            
        if x_finite_dim is not None:
            self.finite_net = VectorFieldMLP(
                input_dim = x_finite_dim,
                x_emb_channels = self.conv_channels,
                ctx_emb_channels = self.ctx_embedding_channels,
                modes = 2*self.modes if self.always_equispaced else 2*self.modes*self.modes,
                time_emb_dim = self.time_embedding_channels,
                cond_emb_dim = 32,
                hidden_features = 64,
                num_layers=1,
                global_mlp_ratio=1,
                num_intermediate_mlp_layers=0,
                adamlp_ratio=1,
                act=act,
            )
            self.x_finite_embedding_net = FiniteXEmbedding(
                input_dim = x_finite_dim,
                num_layers=1,
                hidden_dim=64,
                output_dim=self.ctx_embedding_channels,
                act=act
            )


        self.register_buffer("_positions", None)
        self.register_buffer("_ctx_positions", None)
        self._batch_size = None

        # Initialize output layer to reduce number of convolutional channels to dimension of input channels
        self.f_last = conv_module(self.conv_channels, self.in_channels, 1)

    def get_trafo(self, batch_size:int,point_positions:Tensor, which:str ="x") -> SpectralTransform:
        """
        Return saved transformation for a given set of positions or construct a new one 
        and save it.

        Args:
        batch_size: Batch size of the input
        point_positions: Tensor of shape (batch_size, n_points, 2)
        which: "x" or "ctx", to indicate which transformation we are comparing against.
        """
        # Check if trafo already exists
        positions = self._positions if which == "x" else self._ctx_positions
        if positions is not None and torch.equal(positions, point_positions) and self._batch_size == batch_size:
            return self._trafo if which == "x" else self._ctx_trafo

        else:
            self._batch_size = batch_size
            trafo = VFT2D(batch_size, point_positions.shape[-2], self.modes, point_positions,device=point_positions.device)
            if which == "x":
                self._positions = point_positions
                self._trafo = trafo
            else:
                self._ctx_positions = point_positions
                self._ctx_trafo = trafo
            return trafo       
        
    def process_points(self,
                    x: Tensor, 
                    point_positions: Optional[Tensor],
                    which:str ="x") -> Tuple[Tensor, SpectralTransform]:
        """
        Broadcasts the input and positions to the right shapes, and returns the transform
        for these point positions.

        Args:
        x: Tensor of shape (batch_size, in_channels, n_points) or (batch_size, in_channels, nx_points, ny_points)
        point_positions: Tensor of shape (batch_size, n_points,2) or (n_points,2). If None, assume equispaced points on full domain.
        which: "x" or "ctx", to indicate which input is being processed.
        """
        batch_size = x.shape[0]
        if x.ndim == 3:
            x_n_points = x.shape[-1]
        elif x.ndim == 4:
            x_n_points = x.shape[-2] * x.shape[-1]
        else:
            raise ValueError(f"Input x must be 3D or 4D tensor. Got {x.ndim}D tensor.")
        
        if point_positions is None:
            trafo = FFT2D(modes = self.modes, device = x.device)
            #point_positions are ignored, for now just sample a random tensor of the correct shape
            point_positions = torch.rand((x.shape[0], x_n_points,2), device=x.device, dtype=x.dtype)

        else:
            assert point_positions.ndim == 2 or point_positions.ndim == 3, \
                f"point_positions must have shape (batch_size, n_points,2) or (n_points,2). Got {point_positions.shape}."
            n_points = point_positions.shape[-2]
            assert x_n_points == n_points, \
                f"Number of points in x and positions must be equal. x has {x_n_points} points, point_positions has {n_points} points."
            #if point positions are fixed across batch, we repeat them
            if point_positions.ndim ==2: 
                point_positions = point_positions.unsqueeze(0).expand(batch_size, -1,-1) # (batch_size, n_points,2)
            elif point_positions.shape[0] == 1: # if point positions are fixed across batch, we repeat them
                point_positions = point_positions.expand(batch_size, -1,-1) # (batch_size, n_points,2)

            if self.padding is not None:
                x, point_positions = self.padding(x, point_positions)

            assert isinstance(point_positions, Tensor)
            trafo = self.get_trafo(batch_size, point_positions, which = which)

        return point_positions, trafo

    def forward(self,
                x:Tensor,
                ctx:Tensor,
                t:Tensor,
                x_finite:Optional[Tensor] = None,
                point_positions:Optional[Tensor]=None,
                ctx_point_positions:Optional[Tensor] = None,
                )-> Union[Tensor, Tuple[Tensor, Tensor]]:
        
        """
        Forward pass of the velocity network. We first process the inputs and construct
        the point positional transformations for both x and ctx.

        We predict the velocity from x, ctx, t, the point positions where x and ctx where
        measured (potentially different positions). Optionally, if additional finite conditions
        x_finite are given, we use them to predict the velocity, and additionally use the other
        variables to predict the velocity for x_finite.
        """
        # x has shape (batch, in_channels, n_points) or (batch, in_channels, nx_points, ny_points)
        # ctx has shape (batch, ctx_in_channels, nctx_points) or (batch, ctx_in_channels, nctx_points_x, nctx_points_y)
        # t has shape (batch, 1)
        # x_finite has shape (batch, finite_dim)
        # point_positions can have shape (batch_size, n_points ,2) or (n_points 2) --> for fixed point positions across batch
        # ctx_point_positions can have shape (batch_size, nctx_points ,2) or (nctx_points 2) --> for fixed point positions across batch
        _permute_shape = (0, 1, 2, 3) if x.ndim ==4  else (0, 2, 1)


        _point_positions, trafo = self.process_points(x, point_positions,which="x")
        if ctx_point_positions is None:
            ctx_point_positions = _point_positions
            ctx_trafo = trafo
        else:
            ctx_point_positions, ctx_trafo = self.process_points(ctx, ctx_point_positions,which="ctx")
        point_positions = _point_positions
        t_embedding = self.time_embedding_net(t) # (batch_size, time_embedding_channels)
        p_embedding = self.position_embedding_net(point_positions) if self.position_embedding_channels is not None else None# (batch_size, n_points, position_embedding_channels)
        p_ctx_embedding = self.ctx_position_embedding_net(ctx_point_positions) if self.position_embedding_channels is not None else None # (batch_size, nctx_points, position_embedding_channels)

        if self.padding is not None: 
            x = self.padding(x)
            ctx = self.padding(ctx)
        x = self.x_input_layer(x)
        x = x.permute(*_permute_shape) # (batch_size, n_points, n_channels) or (batch_size, nx_points, ny_points, n_channels)
        ctx = self.ctx_input_layer(ctx)
        ctx = ctx.permute(*_permute_shape) # (batch_size, nctx_points, n_channels) or (batch_size, nctx_points_x, nctx_points_y, n_channels)
        ctx = self.ctx_self_block(ctx, t_embedding, p_ctx_embedding, transform=ctx_trafo)
        x_finite_embedding = self.x_finite_embedding_net(x_finite) if x_finite is not None else None
        for block in self.blocks:
            x = block(x, ctx, t_embedding,  p_embedding, trafo, ctx_trafo, x_finite_embedding)
        
        if x_finite is not None:

            # Combine spectral features of x and ctx, pass with time embedding and x_finite
            # to make velocity prediction for x_finite.
            x_spectral = torch.view_as_real(trafo.forward(x)) #(batch_size, modes, conv_channels,2) or (batch_size, modes, modes, conv_channels,2)
            if x_spectral.ndim == 5:
                x_spectral = x_spectral.reshape(x_spectral.shape[0],-1, self.conv_channels, 2) # (batch_size, modes*modes, conv_channels,2)

            
            x_spectral = x_spectral.permute(0, 2 , 1, 3) # (batch_size, conv_channels, modes, 2) or (batch_size, conv_channels, modes*modes, 2)
            x_spectral = x_spectral.reshape(x_spectral.shape[0],  self.conv_channels, 2 * self.modes) # (batch_size, conv_channels, modes*2) or (batch_size, conv_channels, modes*modes*2)
            ctx_spectral = torch.view_as_real(ctx_trafo.forward(ctx)) #(batch_size, modes, ctx_embedding_channels,2) or (batch_size, modes, modes, ctx_embedding_channels,2)
            if ctx_spectral.ndim == 5:
                ctx_spectral = ctx_spectral.reshape(ctx_spectral.shape[0],-1, self.ctx_embedding_channels, 2) # (batch_size, modes*modes, ctx_embedding_channels,2)
            ctx_spectral = ctx_spectral.permute(0, 2 , 1, 3) # (batch_size, conv_channels, modes, 2) or (batch_size, conv_channels, modes*modes, 2)
            ctx_spectral = ctx_spectral.reshape(ctx_spectral.shape[0],  self.ctx_embedding_channels, 2 * self.modes) # (batch_size, conv_channels, modes*2) or (batch_size, conv_channels, modes*modes*2)
            x_finite_pred = self.finite_net(x_finite, x_spectral,ctx_spectral, t_embedding)
        
        x = x.permute(*_permute_shape) # (batch_size, conv_channels, n_points)
        x = self.f_last(x)
        if self.padding is not None:
            x = self.padding.unpad(x)

        if x_finite is not None:
            return x, x_finite_pred
        else:
            return x
        

class FNOPE_2D(nn.Module):
    #adapted from https://github.com/sbi-dev/sbi/blob/main/sbi/neural_nets/estimators/flowmatching_estimator.py
    def __init__(
        self,
        x: Tensor,
        ctx: Tensor,
        simulation_grid: Optional[Tensor] = None,
        x_finite: Optional[Tensor] = None,
        modes: int = 32,
        conv_channels: int = 16,
        ctx_embedding_channels: int = 16,
        time_embedding_channels: int = 8,
        position_embedding_channels: Optional[int] = 8,
        num_layers:int =5,
        act=nn.GELU(),
        base_dist="gp",
        base_dist_lengthscale_multiplier: Optional[Union[float, str]] = None,
        noise_scale = 1e-5,
        padding = {"type":"zero","pad_length":10},
        training_point_noise = {"jitter": 1e-3, "target_gridsize": 256},
        always_equispaced: bool = False,
        always_match_x_theta: bool = False,
        **kwargs,
    ):
        """
        Create a flow matching model for given input and context. Infer dimensionalities and standardization from the inputs directly.

        Args:
        x: Tensor of shape (batch_size, in_channels, n_points)
        ctx: Tensor of shape (batch_size, ctx_in_channels, nctx_points)
        simulation_grid: Tensor of shape (n_points). If None, we assume an equispaced grid on the full domain. Used for standardization.
        x_finite: Optional tensor of shape (batch_size, finite_dim)
        modes: Number of spectral modes to use for the FNO blocks
        conv_channels: Number of convolutional channels to use for the FNO blocks
        ctx_embedding_channels: Number of channels to use for the context embedding
        time_embedding_channels: Number of channels to use for the time embedding
        position_embedding_channels: Number of channels to use for the position embedding
        num_layers: Number of FNO blocks to use
        act: Activation function to use for the FNO blocks
        base_dist: Base distribution to use for the flow matching model. This is the distribution at flow time T=1. By default we use a Gaussian Process with a similar spectrogram as the input data.
        noise_scale: Minimum time to sample for loss (for numerical stability)
        padding: Arguments for how to pad the inputs in the FNO blocks (if at all)
        training_point_noise: Jitter to add to the point positions for training. This is used to sample new points for the loss function.
        always_equispaced: If True, we always use equispaced points for the input and context. 
                           This will force the net to only use FFT, and never the VFFT - will raise a warning if we pass in point_positions.
        always_match_x_theta: If True, we always use the same point positions for x and ctx.
                            This will raise a warning if we pass in ctx_positions.
        """
        
        super().__init__()
        self.modes = modes
        self.always_equispaced = always_equispaced

        self.in_channels, self.input_shape, self.x_standardizing_net = self._read_continuous_input(x, simulation_grid=simulation_grid, standardize=True)
        self.ctx_in_channels, self.nctx_points,self.ctx_standardizing_net = self._read_continuous_input(ctx, simulation_grid=simulation_grid, standardize=True)
        self.x_finite_dim, self.x_finite_standardizing_net = self._read_finite_input(x_finite, standardize=True)
        
        
        self.conv_channels = conv_channels
        self.ctx_embedding_channels = ctx_embedding_channels
        self.time_embedding_channels = time_embedding_channels
        self.position_embedding_channels = position_embedding_channels
       
        self.num_layers = num_layers
        self.training_point_jitter = training_point_noise.pop("jitter", 1e-3)
        self.target_gridsize = training_point_noise.pop("target_gridsize", 256)

        
        # Select base distribution
        if base_dist == "gp":
            if base_dist_lengthscale_multiplier is None:
                target_freq = (self.modes//2+1)/4.0
            elif isinstance(base_dist_lengthscale_multiplier, str):
                target_freq = (self.modes//2+1)/4.0
            else:
                target_freq = self.modes * base_dist_lengthscale_multiplier
            domain_size = math.floor(math.sqrt(x.shape[-1])) if x.ndim == 3 else x.shape[-2]
            self.base_dist = FrequencyThresholdedGaussianProcess2d(target_freq, self.in_channels, domain_size, default_2d = always_equispaced)
        elif base_dist == "wn":
            if self.always_equispaced:
                domain_size = math.floor(math.sqrt(x.shape[-1])) if x.ndim == 3 else x.shape[-2]
            else:
                domain_size = x.shape[-1] if x.ndim == 3 else x.shape[-2] * x.shape[-1]
            self.base_dist = WhiteNoise(self.in_channels, domain_size,default_2d=self.always_equispaced)
        else:
            raise NotImplementedError(f"Base distribution {base_dist} not implemented.")
        
        if self.always_equispaced:
            if self.position_embedding_channels is not None:
                warn("Position embedding is currently not supported for equispaced points. Ignoring position embedding.")
                self.position_embedding_channels = None

        else:
            if padding is not None and padding["type"].lower() != "none":
                warn("Padding is not supported for non-equispaced points. Ignoring padding.")
                padding = {"type":"none","pad_length":0} # no padding for non-equispaced points
        self.always_match_x_theta = always_match_x_theta
        
        self.vel_net = Velocity_Unified2D(modes=self.modes,
                                        in_channels=self.in_channels,
                                        conv_channels=self.conv_channels,
                                        ctx_in_channels=self.ctx_in_channels,
                                        x_finite_dim=self.x_finite_dim,
                                        ctx_embedding_channels=self.ctx_embedding_channels,
                                        time_embedding_channels=self.time_embedding_channels,
                                        position_embedding_channels=self.position_embedding_channels,   
                                        num_layers=self.num_layers,
                                        equispaced = True,
                                        padding=padding,
                                        act=act,
                                        always_equispaced=self.always_equispaced,)
        
        self.noise_scale = noise_scale

    def _read_continuous_input(self,input:Tensor, simulation_grid:Optional[Tensor]=None, standardize:bool = True):
        """
        Shape handling for the input tensors and returning standardizing nets

        Args:
        input: Tensor of shape (batch_size, in_channels, nx_points, ny_points) or (batch_size, in_channels, n_points)
        simulation_grid: Tensor of shape (n_points,2) or (nx_points,ny_points,2). If None, we assume an equispaced grid on the full domain.
        standardize: bool, whether to standardize the input or not.
        """
        if self.always_equispaced:
            assert simulation_grid is None, "Simulation grid must be None if always_equispaced is True."
            assert input.ndim == 4, f"For equispaced data, expected input of shape (batch_size, in_channels, nx_points, ny_points). Got {input.shape} instead."
            in_channels = input.shape[1]
            input_shape = (input.shape[2], input.shape[3])

        else:
            assert input.ndim == 3 or input.ndim == 4, f"Input must be 3D or 4D tensor. Got {input.ndim}D tensor."
            if input.ndim == 4:
                warn("Input is 4D tensor. Assuming input is of shape (batch_size, in_channels, nx_points, ny_points)." \
                "Reshaping to (batch_size, in_channels, n_points) with n_points = nx_points * ny_points.")
                in_channels = input.shape[1]
                input_shape = (input.shape[2], input.shape[3])
                n_points = input.shape[2] * input.shape[3]
                input = input.reshape(input.shape[0], in_channels, n_points)
            elif input.ndim == 3:
                n_points = input.shape[2]
                input_shape = (input.shape[2], 1)
                in_channels = input.shape[1]

            if simulation_grid is not None:
                simulation_grid = self._reshape_point_positions(simulation_grid)



        if standardize:
            standardizing_net = FilterStandardizing2d(input,simulation_grid,num_channels=in_channels,modes=self.modes,cutoff=False)
        else:
            standardizing_net = IdentityStandardizing()
        return in_channels, input_shape, standardizing_net
    
    def _read_finite_input(self,input:Optional[Tensor], standardize:bool = True):
        """
        Shape handling for the finite input tensors and returning standardizing nets

        Args:
        input: Tensor of shape (batch_size, finite_dim)
        standardize: bool, whether to standardize the input or not.
        """
        if input is None:
            return None, None
        
        assert input.ndim ==2 or input.ndim ==1, f"Finite parameters must be 1D or 2D tensor. Got {input.ndim}D tensor."
        if input.ndim == 1:
            warn("Finite parameters are 1D tensor. Assuming 1 finite parameter, reshaping to (batch_size,1).")
            input = input.unsqueeze(1)
        finite_dim = input.shape[1]
        if standardize:
            standardizing_net = FiniteStandardizing(input)
        else:
            standardizing_net = IdentityStandardizing()
        return finite_dim, standardizing_net
    
    def _reshape_point_positions(self, point_positions:Tensor):
        """
        Given point positions of shape (n_points, 2) or (nx_points, ny_points, 2)
        Make sure point positions are of shape (n_points,2).
        """

        if point_positions.ndim == 3:
            #we are given (nx_points, ny_points, 2)
            point_positions = point_positions.reshape(point_positions.shape[0]*point_positions.shape[1],2)

        return point_positions


    def forward(self,
                x:Tensor,
                ctx:Tensor,
                t:Union[Tensor,float],
                x_finite:Optional[Tensor]=None,
                point_positions:Optional[Tensor] = None,
                ctx_point_positions:Optional[Tensor] = None,
                ):
        
        if self.always_equispaced and point_positions is not None:
            warn("The model was defined with `always_equispaced = True." \
            "However, point positions were still passed. The point positions will"\
            "be ignored.")
            point_positions = None
        if self.always_equispaced and ctx_point_positions is not None:
            warn("The model was defined with `always_equispaced = True." \
            "However, ctx point positions were still passed. The ctx point positions will"\
            "be ignored.")
            ctx_point_postions = None
        if self.always_match_x_theta and ctx_point_positions is not None:
            warn("The model was defined with `always_match_x_theta = True." \
            "However, ctx point positions were still passed. The ctx point positions will"\
            "be assumed to be the same as `point_positions`.")
            ctx_point_positions = point_positions


        # Ensure all shapes are correct before passing to the velocity network.
        x_batch_size = x.shape[0]
        ctx_batch_size = ctx.shape[0]
        if x_batch_size == 1:
            x = x.expand(ctx_batch_size, *x.shape[1:])
        elif ctx_batch_size == 1:
            ctx = ctx.expand(x_batch_size, *ctx.shape[1:])
        if isinstance(t, float):
            t = torch.Tensor([t]).to(x.device)
            t = t.expand(x.shape[0])
        else:
            assert isinstance(t,Tensor)
            if t.shape == ():
                t = t.to(x.device)
                t = t.expand(x.shape[0])


        return self.vel_net(x=x, ctx=ctx, t=t,x_finite=x_finite, point_positions=point_positions, ctx_point_positions=ctx_point_positions)


    def _vft_loss(self,
                  x:Tensor,
                  ctx:Tensor,
                  x_finite:Optional[Tensor]=None,
                  simulation_positions:Optional[Tensor]=None,
                  ctx_simulation_positions:Optional[Tensor]=None,
                  **kwargs):
        
        """
        See self.loss() for docstring.
        """
        # sample random times in [0,1]
        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        tf = t[..., None] #different shape broadcasting for x_finite
        t_ = t[..., None,None]

        # broadcast positions to correct shape
        if simulation_positions is None:
            assert x.ndim ==4 and ctx.ndim == 4, f"Cannot infer simulation positions from tensors of shape (batch_size,n_channels,n_points).\n" \
                                                  "Please specify the points explicitly or pass a 4D tensor."
            x_positions = torch.linspace(0, 1, x.shape[-2], device=x.device, dtype=x.dtype).repeat_interleave(x.shape[-1])
            y_positions = torch.linspace(0, 1, x.shape[-1], device=x.device, dtype=x.dtype).repeat(x.shape[-2])
            old_positions = torch.stack([x_positions, y_positions], dim=-1)
        elif isinstance(simulation_positions, Tensor):
            old_positions = self._reshape_point_positions(simulation_positions)

        #expand batch size.
        old_positions.unsqueeze(0) # (1, n_points, 2)
        old_positions = old_positions.expand(x.shape[0], -1, -1)
        if ctx_simulation_positions is None:
            old_ctx_positions = old_positions
        elif isinstance(ctx_simulation_positions, Tensor):
            old_ctx_positions = self._reshape_point_positions(ctx_simulation_positions)
            old_ctx_positions.unsqueeze(0) # (1, n_points, 2)
            old_ctx_positions = old_ctx_positions.expand(ctx.shape[0], -1, -1)

        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
            ctx = ctx.reshape(ctx.shape[0], ctx.shape[1], -1)




        x = self.x_standardizing_net.standardize(x,point_positions=old_positions)
        ctx = self.ctx_standardizing_net.standardize(ctx,point_positions=old_ctx_positions)
        if self.x_finite_standardizing_net is not None and x_finite is not None:
            x_finite = self.x_finite_standardizing_net.standardize(x_finite)

        # Currently, the shapes are:
        #old_positions (batch_size, nsim_points, 2)
        #new_positions (batch_size, ninterp_points,2)
        #new_ctx_positions (batch_size, ninterp_points,2)
        #x (batch_size, in_channels, nsim_points)
        #ctx (batch_size, ctx_in_channels, nsim_points)
        #x_finite (batch_size, finite_dim)
     
        with torch.no_grad():

            position_jitter = torch.randn_like(old_positions, device=old_positions.device, dtype=old_positions.dtype) * self.training_point_jitter
            new_positions = old_positions + position_jitter

            epsilon_positions = old_positions
            epsilon = self.base_dist.sample_like(x, sampling_positions=epsilon_positions[0])


            n_points = old_positions.shape[1]
            # Create a mask for each set of positions in old_positions
            batch_size = old_positions.shape[0]
            mask = torch.zeros((batch_size, n_points), dtype=torch.bool, device=old_positions.device)
            target_gridsize = min(self.target_gridsize, n_points)
            indices = torch.rand(batch_size, n_points, device=old_positions.device).argsort(dim=-1)[:, :target_gridsize]
            mask.scatter_(1, indices, True)

            # Apply the mask to select new positions
            new_positions = old_positions[mask].view(batch_size, target_gridsize, -1)
            mask = mask.unsqueeze(1)
            x =x[mask.expand(-1,self.in_channels,-1)].view(batch_size, self.in_channels, target_gridsize)
            epsilon = epsilon[mask.expand(-1,self.in_channels,-1)].view(batch_size, self.in_channels, target_gridsize)
            # epsilon = self.base_dist.sample_like(x,sampling_positions=new_positions)

            if self.always_match_x_theta:
                #always match x and theta, so always interpolate onto the same grid when training
                new_ctx_positions = new_positions
                ctx = ctx[mask.expand(-1,self.ctx_in_channels,-1)].view(batch_size, self.ctx_in_channels, target_gridsize)
            else:
                n_ctx_points = old_ctx_positions.shape[1]
                position_jitter = torch.randn_like(old_ctx_positions, device=old_ctx_positions.device, dtype=old_ctx_positions.dtype) * self.training_point_jitter
                new_ctx_positions = old_ctx_positions + position_jitter
                mask = torch.zeros((batch_size, n_ctx_points), dtype=torch.bool, device=old_ctx_positions.device)
                ctx_target_gridsize = min(self.target_gridsize, n_ctx_points)
                indices = torch.rand(batch_size, n_ctx_points, device=old_ctx_positions.device).argsort(dim=-1)[:, :ctx_target_gridsize]
                mask.scatter_(1, indices, True)
                # Apply the mask to select new positions
                new_ctx_positions = old_ctx_positions[mask].view(batch_size, ctx_target_gridsize, -1)
                mask = mask.unsqueeze(1).expand(-1, self.ctx_in_channels, -1)
                ctx = ctx[mask].view(batch_size, self.ctx_in_channels, ctx_target_gridsize)



            # new_positions = torch.rand((x.shape[0], self.target_gridsize,2), device=x.device, dtype=x.dtype)
            # if self.always_match_x_theta:
            #     #always match x and theta, so always interpolate onto the same grid when training
            #     new_ctx_positions = new_positions
            # else:
            #     new_ctx_positions = torch.rand((ctx.shape[0], self.target_gridsize,2), device=ctx.device, dtype=ctx.dtype)

            # #Mimic interpolation of x by sampling epsilon on different grid and interpolating
            # #Need to make sure this is called before x is interpolated to new grid
            # # eps_x_positions = torch.linspace(0, 1, 40, device=x.device, dtype=x.dtype).repeat_interleave(40)
            # # eps_y_positions = torch.linspace(0, 1, 40, device=x.device, dtype=x.dtype).repeat(40)
            # # epsilon_positions = torch.stack([eps_x_positions, eps_y_positions], dim=-1)
            # # epsilon_positions = epsilon_positions.unsqueeze(0).expand(x.shape[0],-1, -1)

            # epsilon_positions = old_positions
            # epsilon = self.base_dist.sample_like(x,sampling_positions=epsilon_positions[0])

            # # Also sample and interpolate epsilon to the new grid
            # epsilon = self.interpolator.interpolate2d(epsilon.permute(0, 2, 1), epsilon_positions, new_positions).permute(0, 2, 1)

            # # alternative, sample epsilon directly on the new grid
            # # epsilon = self.base_dist.sample_like(x,sampling_positions=new_positions)


            # # x_upsampled = torch.nn.functional.interpolate(x,size=5,mode="linear", align_corners=False)
            # # ctx_upsampled = torch.nn.functional.interpolate(ctx,size=5,mode="linear", align_corners=False)
            # # upsampled_positions = torch.nn.functional.interpolate(old_positions,size=5,mode="linear", align_corners=False)

            # # x = self.interpolator.interpolate2d(x_upsampled.permute(0, 2, 1), upsampled_positions, new_positions).permute(0, 2, 1)
            # # ctx = self.interpolator.interpolate2d(ctx_upsampled.permute(0, 2, 1), upsampled_positions, new_ctx_positions).permute(0, 2, 1)

            # # x = self.interpolator.interpolate2d(x.permute(0, 2, 1), old_positions, new_positions).permute(0, 2, 1)
            # # ctx = self.interpolator.interpolate2d(ctx.permute(0, 2, 1), old_positions, new_ctx_positions).permute(0, 2, 1)


        
        x_prime = (1 - t_) * x + (t_ + self.noise_scale) * epsilon
        vector_field = epsilon - x

        # if x_finite is none, we can ignore it in the loss and in the prediction.
        if x_finite is None:
            pred = self.forward(x_prime, ctx, t, None, point_positions=new_positions,ctx_point_positions=new_ctx_positions)
            # compute the mean squared error between the vector fields
            mse = torch.mean(
                (pred - vector_field) ** 2, dim=(0,1)
            )
        #Otherwise, we use it to predict the continuous x and vice versa, and then average
        #their respective losses for the final loss.
        else:
            finite_epsilon = torch.randn_like(x_finite, device=x_finite.device, dtype=x_finite.dtype)
            x_finite_prime = (1 - tf) * x_finite + (tf + self.noise_scale) * finite_epsilon
            finite_vector_field = finite_epsilon - x_finite
            x_pred, x_finite_pred = self.forward(x_prime, ctx, t, x_finite_prime, point_positions=new_positions,ctx_point_positions=new_ctx_positions)
            
            # Calculate the continuous and finite MSE separately, otherwise x_finite mse not
            # equally weighted
            x_mse = torch.mean(
                (x_pred - vector_field) ** 2, dim=(1,2)
            )
            finite_mse = torch.mean(
                (x_finite_pred - finite_vector_field) ** 2, dim=1
            )
            mse = x_mse+finite_mse
            
        return mse.mean()
    
    def _fft_loss(self,
                  x:Tensor,
                  ctx:Tensor,
                  x_finite:Optional[Tensor]=None,
                  **kwargs):
        """
        See self.loss() for docstring.
        """
        
        # sample random times in [0,1]
        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        tf = t[..., None] #different shape broadcasting for x_finite
        t_ = t[..., None,None,None]

        x = self.x_standardizing_net.standardize(x,point_positions=None)
        ctx = self.ctx_standardizing_net.standardize(ctx,point_positions=None)
        if self.x_finite_standardizing_net is not None and x_finite is not None:
            x_finite = self.x_finite_standardizing_net.standardize(x_finite)

        # Currently, the shapes are:
        #x (batch_size, in_channels, nx_points, ny_points)
        #ctx (batch_size, ctx_in_channels, nx_points,ny_points)
        #x_finite (batch_size, finite_dim)

        eps_x_positions = torch.linspace(0, 1, x.shape[-2], device=x.device, dtype=x.dtype).repeat_interleave(x.shape[-1])
        eps_y_positions = torch.linspace(0, 1, x.shape[-1], device=x.device, dtype=x.dtype).repeat(x.shape[-2])
        epsilon_positions = torch.stack([eps_x_positions, eps_y_positions], dim=-1)
        epsilon = self.base_dist.sample_like(x, sampling_positions=epsilon_positions).view(*x.shape)
        x_prime = (1 - t_) * x + (t_ + self.noise_scale) * epsilon
        vector_field = epsilon - x

        # if x_finite is none, we can ignore it in the loss and in the prediction.
        if x_finite is None:
            pred = self.forward(x_prime, ctx, t, None, point_positions=None,ctx_point_positions=None)
            # compute the mean squared error between the vector fields
            mse = torch.mean(
                (pred - vector_field) ** 2, dim=(0,1)
            )
        #Otherwise, we use it to predict the continuous x and vice versa, and then average
        #their respective losses for the final loss.
        else:
            finite_epsilon = torch.randn_like(x_finite, device=x_finite.device, dtype=x_finite.dtype)
            x_finite_prime = (1 - tf) * x_finite + (tf + self.noise_scale) * finite_epsilon
            finite_vector_field = finite_epsilon - x_finite
            x_pred, x_finite_pred = self.forward(x_prime, ctx, t, x_finite_prime, point_positions=None,ctx_point_positions=None)
            
            # Calculate the continuous and finite MSE separately, otherwise x_finite mse not
            # equally weighted
            x_mse = torch.mean(
                (x_pred - vector_field) ** 2, dim=(1,2,3)
            )
            finite_mse = torch.mean(
                (x_finite_pred - finite_vector_field) ** 2, dim=1
            )
            mse = x_mse+finite_mse
            
        return mse.mean()

    def loss(self,
            x:Tensor,
            ctx:Tensor,
            x_finite:Optional[Tensor] = None,
            simulation_positions: Optional[Tensor] = None,
            ctx_simulation_positions: Optional[Tensor] = None,
            **kwargs):
        """
        Calculates the flow matching loss of input x conditioned on ctx. Optionally,
        also computes the flow matching loss of an additional, finite (non-functional)
        input x_finite conditioned on ctx.

        Args:
        x: Tensor of shape (batch_size, in_channels, n_points) or (batch_size, in_channels, nx_points, ny_points)
        ctx: Tensor of shape (batch_size, ctx_in_channels, n_ctx_points) or (batch_size, ctx_in_channels, nx_ctx_points, ny_ctx_points)
        x_finite: Optional tensor of shape (batch_size, finite_dim)
        simulation_positions: Optional tensor of shape (n_points,2) or (nx_points,ny_points,2).
            If not provided, the simulation positions are set to equispaced points in 
            [0,1] of length n_points.
        """

        if self.always_equispaced:
            if simulation_positions is not None:
                warn("The model was defined with `always_equispaced = True." \
                "However, simulation positions were still passed. The simulation positions will"\
                "be ignored.")
                simulation_positions = None
            if ctx_simulation_positions is not None:
                warn("The model was defined with `always_equispaced = True." \
                "However, ctx simulation positions were still passed. The ctx simulation positions will"\
                "be ignored.")
                ctx_simulation_positions = None
            return self._fft_loss(x, ctx, x_finite, simulation_positions=simulation_positions, **kwargs)
        else:
            if self.always_match_x_theta and ctx_simulation_positions is not None:
                warn("The model was defined with `always_match_x_theta = True." \
                "However, ctx simulation positions were still passed. The ctx simulation positions will"\
                "be assumed to be the same as `simulation_positions`.")
                ctx_simulation_positions = simulation_positions
            return self._vft_loss(x, ctx, x_finite, simulation_positions=simulation_positions,ctx_simulation_positions=ctx_simulation_positions, **kwargs)

    def _wrap_state(self, x:Tensor, x_finite:Tensor):
        """
        Wrap x and x_finite together, used for constructing zuko flow.
        """
        assert self.x_finite_dim is not None, "x_finite_dim must be set to wrap state."
        flattented_x = x.reshape(x.shape[0], -1)
        state = torch.cat([flattented_x,x_finite],dim=-1)
        state = state.unsqueeze(1)
        return state
    
    def _unwrap_state(self, state:torch.Tensor, unwrapped_shape:Tuple[int]):
        """
        Separate state into x and x_finite, used for interpreting joint zuko flows.
        """
        assert self.x_finite_dim is not None, "x_finite_dim must be set to unwrap state."
        x = state[..., :-self.x_finite_dim]
        x = x.reshape(x.shape[0], *unwrapped_shape)
        x_finite = state[..., -self.x_finite_dim:]
        x_finite = x_finite.reshape(x.shape[0], self.x_finite_dim)
        return x,x_finite

    @torch.no_grad()
    def unnormalized_log_prob(self,
                x: Tensor,
                ctx: Tensor,
                x_finite: Optional[Tensor]=None,
                point_positions: Optional[Tensor]=None,
                ctx_point_positions: Optional[Tensor]= None,
                **kwargs):
        
        """
        Calculates the (unnormalized) log probability of the input x conditioned on ctx. 
        Optionally, also computes the log probability of an additional, finite (non-functional)
        input x_finite conditioned on ctx.
        """
        if self.always_equispaced and (point_positions is not None or ctx_point_positions is not None):
            warn("The model was defined with `always_equispaced = True." \
                "However, point positions or ctx_point_positions were still passed." \
                "The point positions willbe ignored.")
            point_positions = None
            ctx_point_positions = None
        elif not self.always_equispaced and ctx_point_positions is None:
            ctx_point_positions = point_positions

        
        if point_positions is None:
            unwrapped_shape = (self.in_channels, self.input_shape[0], self.input_shape[1])
        elif point_positions.ndim == 3:
            unwrapped_shape = (self.in_channels, point_positions.shape[0], point_positions.shape[1])
            point_positions = self._reshape_point_positions(point_positions)
        else:
            unwrapped_shape = (self.in_channels, point_positions.shape[0])

        if ctx_point_positions is not None:
            if ctx_point_positions.ndim == 3:
                ctx_point_positions = self._reshape_point_positions(ctx_point_positions)

        if not self.always_equispaced:
            x = x.reshape(*x.shape[:-2],-1)
            ctx = ctx.reshape(*ctx.shape[:-2], -1)

        x = self.x_standardizing_net.standardize(x,point_positions=point_positions)
        ctx = self.ctx_standardizing_net.standardize(ctx,point_positions=ctx_point_positions)
        if self.x_finite_standardizing_net is not None and x_finite is not None:
            x_finite = self.x_finite_standardizing_net.standardize(x_finite)



        if x_finite is not None:
            state = self._wrap_state(x,x_finite)
        else:
            state = x
        log_probs = self.flow(ctx=ctx,point_positions=point_positions,ctx_point_positions=ctx_point_positions,unwrapped_shape=unwrapped_shape,**kwargs).log_prob(state)
        return log_probs

    @torch.no_grad()
    def sample(self,
            num_samples:int,
            ctx:Tensor,
            point_positions:Optional[Tensor]=None,
            ctx_point_positions:Optional[Tensor]=None,
            **kwargs):
        """
        Samples from the conditional flow matching model, with batch_dims = sample_shape.

        Args:
        num_samples: Number of samples to draw from the model.
        ctx: Tensor of shape (ctx_in_channels, nctx_points) or (ctx_in_channels, nx_ctx_points, ny_ctx_points)
        point_positions: Tensor of shape (n_points,2) or (nx_points,ny_points,2). If None, assume equispaced points on full domain.
                         We do not allow a batch of point positions, as we are only sampling on the user provided positions.
        ctx_point_positions: Tensor of shape (batch_size, nctx_points) or (nctx_points). If None, assume equispaced points on full domain.
        **kwargs: Additional arguments to pass to the flow.
        
        """
        ctx = ctx.unsqueeze(0)
        ctx = ctx.expand(num_samples, *(-1 for i in range(len(ctx.shape[1:])))) # (num_samples, ctx_in_channels, nctx_points) or (num_samples, ctx_in_channels, nx_ctx_points, ny_ctx_points)
        ctx = self.ctx_standardizing_net.standardize(ctx,point_positions=ctx_point_positions)
        if not self.always_equispaced:
            ctx = ctx.reshape(ctx.shape[0], ctx.shape[1], -1)

        if self.always_equispaced and (point_positions is not None or ctx_point_positions is not None):
            warn("The model was defined with `always_equispaced = True." \
                 "However, point positions or ctx_point_positions were still passed." \
                 "The point positions willbe ignored.")
            point_positions = None
            ctx_point_positions = None
        elif not self.always_equispaced and ctx_point_positions is None:
            ctx_point_positions = point_positions
        
        if point_positions is None:
            unwrapped_shape = (self.in_channels, self.input_shape[0], self.input_shape[1])
        elif point_positions.ndim == 3:
            unwrapped_shape = (self.in_channels, point_positions.shape[0], point_positions.shape[1])
            point_positions = self._reshape_point_positions(point_positions)
        else:
            unwrapped_shape = (self.in_channels, point_positions.shape[0])

        if ctx_point_positions is not None:
            if ctx_point_positions.ndim == 3:
                ctx_point_positions = self._reshape_point_positions(ctx_point_positions)
        samples = self.flow(ctx=ctx,point_positions=point_positions,ctx_point_positions=ctx_point_positions,unwrapped_shape=unwrapped_shape,**kwargs).sample((num_samples,))
        if self.x_finite_dim is not None:
            samples, x_finite = self._unwrap_state(samples,unwrapped_shape)
            samples = self.x_standardizing_net.unstandardize(samples,point_positions=point_positions)
            if self.x_finite_standardizing_net is not None:
                x_finite = self.x_finite_standardizing_net.unstandardize(x_finite)
            return samples, x_finite
        else:
            samples = self.x_standardizing_net.unstandardize(samples,point_positions=point_positions)
            samples = samples.reshape(-1,*unwrapped_shape)
            return samples

    def sample_and_log_prob(
        self,
        sample_shape:Union[torch.Size,Tuple[int]],
        ctx:Tensor,
        point_positions=Optional[Tensor],
        ctx_point_positions=Optional[Tensor],
        **kwargs
    ):
        # samples, log_probs = self.flow(ctx, point_positions=point_positions,ctx_point_positions=ctx_point_positions,**kwargs).rsample_and_log_prob(sample_shape)
        # return samples, log_probs
        raise NotImplementedError("Log probability not implemented yet. Need to first verify it is correct")

    def flow(self,
            ctx:Tensor,
            point_positions:Optional[Tensor]=None,
            ctx_point_positions:Optional[Tensor]=None,
            unwrapped_shape:Optional[Tuple[int]]=None,
            atol:float=1e-2,
            rtol:float=1e-2,
            exact:bool=False):
        """
        Construct a zuko Continuous Normalizing Flow with the velocity estimator.
        This is used for sampling and log probability calculations.

        See `self.sample()` for arguments.
        Args:
        unwrapped_shape: Shape of the unwrapped state. This is used to reshape the state
        atol: Absolute tolerance for the ODE solver
        rtol: Relative tolerance for the ODE solver
        """
        if self.x_finite_dim is None:
            transform = zuko.transforms.ComposedTransform(
                FreeFormJacobianTransform(
                    f=lambda t, x: self.forward(x, ctx, t, point_positions=point_positions,ctx_point_positions= ctx_point_positions),
                    t0=ctx.new_tensor(0.0),
                    t1=ctx.new_tensor(1.0),
                    phi=(ctx, *self.vel_net.parameters()),
                    atol=atol,
                    rtol=rtol,
                    exact=exact,
                ),
            )

            base_distribution = self.base_dist.get_pytorch_distribution(sampling_positions=point_positions,device=ctx.device)

            return NormalizingFlow(
                transform=transform,
                base=base_distribution,
            )
        
        else:
            # This is a bit tricky, as zuko treats the state as a single tensor, but we have 
            # the continuous and finite parameters separately.
            #Need to wrap the function to take in the combination of the functional and finite inputs

            def combined_velocity(state, ctx, t,point_positions, ctx_point_positions):
                # unwrap combined state, pass through forward, wrap back for zuko.

                x, x_finite = self._unwrap_state(state,unwrapped_shape=unwrapped_shape) 
                vel_x, vel_x_finite =  self.forward(x, ctx, t, x_finite, point_positions=point_positions,ctx_point_positions=ctx_point_positions)

                vel_state = self._wrap_state(vel_x, vel_x_finite)
                return vel_state
            transform = zuko.transforms.ComposedTransform(
                FreeFormJacobianTransform(
                    f=lambda t, state: combined_velocity(state, ctx, t, point_positions= point_positions,ctx_point_positions=ctx_point_positions),
                    t0=ctx.new_tensor(0.0),
                    t1=ctx.new_tensor(1.0),
                    phi=(ctx, *self.vel_net.parameters()),
                    atol=atol,
                    rtol=rtol,
                    exact=exact,
                ),
            )


            # The noise distribution at T=1 is whatever we define for `base_dist`, and independently sample from the unit Gaussian
            # for x_finite. `zuko.distributions.Joint` combines the two distributions independently and does some shape handling for us.
            cont_dist = self.base_dist.get_pytorch_distribution(sampling_positions=point_positions,device=ctx.device)
            finite_dist = DiagNormal(loc = torch.zeros(self.x_finite_dim,device=ctx.device), scale = torch.ones(self.x_finite_dim,device=ctx.device))
            base_distribution = Joint(cont_dist,finite_dist)

            return NormalizingFlow(
                transform=transform,
                base=base_distribution,
            )