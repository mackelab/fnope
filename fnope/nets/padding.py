import torch
import torch.nn as nn
from typing import Optional

def get_padding(type="zero", pad_length=0):
    if pad_length == 0 or type == 'none':
        return IdentityPad()
    type = type.lower()
    if type == 'zero':
        return ZeroPad(pad_length)
    elif type == 'circular':
        return CircularPad(pad_length)
    elif type == 'reflect':
        return ReflectionPad(pad_length)
    elif type == 'replicate':
        return ReplicationPad(pad_length)
    else:
        raise ValueError(f"Invalid padding type: {type}")
    

#Currently never used as we don't use padding for nonuniform discretizations.
def pad_point_positions(positions:torch.Tensor,pad_length:int):
    """
    Extrapolate the domain from [0,1] to [-extend,1+extend]

    Args:
        positions (Tensor): input positions of shape (batch, n_points)
        pad_length (int): length of padding
    """
    assert positions.ndim == 2, f"Input positions should be 2D tensor of shape (batch_size,n_points), got {positions.ndim}D tensor"
    batchsize, n_points = positions.shape
    points_min = positions.min(dim=-1)[0]
    points_max = positions.max(dim=-1)[0]
    points_range = points_max - points_min
    padding_left = -(torch.arange(pad_length, 0,step=-1, device=positions.device).unsqueeze(0).expand(batchsize,-1)) *points_range.unsqueeze(-1)*(1/n_points)
    padding_right = (torch.arange(0, pad_length, device=positions.device).unsqueeze(0).expand(batchsize,-1)) *points_range.unsqueeze(-1)*(1/n_points) + 1.0

    positions = torch.cat([padding_left, positions, padding_right], dim=-1)

    return positions
    

class IdentityPad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor, point_positions:Optional[torch.Tensor] = None):
        if point_positions is not None:
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions:Optional[torch.Tensor] = None):
        if point_positions is not None:
            return x, point_positions
        return x

class ZeroPad(nn.Module):
    def __init__(self, pad_length):
        super().__init__()
        self.pad_length = pad_length

    def forward(self, x:torch.Tensor, point_positions:Optional[torch.Tensor] = None):
        assert x.shape[-1] // 2 >= self.pad_length, f"Input length is too short for padding: {x.shape[-1]}"
        x = torch.nn.functional.pad(x, (self.pad_length, self.pad_length), mode='constant', value=0)
        if point_positions is not None:
            point_positions = pad_point_positions(point_positions, self.pad_length)
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions:Optional[torch.Tensor] = None):
        x= x[..., self.pad_length:-self.pad_length]
        if point_positions is not None:
            point_positions = point_positions[..., self.pad_length:-self.pad_length]
            return x, point_positions
        return x
    
class CircularPad(nn.Module):
    def __init__(self, pad_length):
        super().__init__()
        self.pad_length = pad_length

    def forward(self, x:torch.Tensor, point_positions:Optional[torch.Tensor] = None):
        assert x.shape[-1] // 2 >= self.pad_length, f"Input length is too short for padding: {x.shape[-1]}"
        x = torch.nn.functional.pad(x, (self.pad_length, self.pad_length), mode='circular')
        if point_positions is not None:
            point_positions = pad_point_positions(point_positions, self.pad_length)
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions:Optional[torch.Tensor] = None):
        x= x[..., self.pad_length:-self.pad_length]
        if point_positions is not None:
            point_positions = point_positions[..., self.pad_length:-self.pad_length]
            return x, point_positions
        return x
    
class ReflectionPad(nn.Module):
    def __init__(self, pad_length):
        super().__init__()
        self.pad_length = pad_length

    def forward(self, x:torch.Tensor, point_positions:Optional[torch.Tensor] = None):
        assert x.shape[-1] // 2 >= self.pad_length, f"Input length is too short for padding: {x.shape[-1]}"
        x = torch.nn.functional.pad(x, (self.pad_length, self.pad_length), mode='reflect')
        if point_positions is not None:
            point_positions = pad_point_positions(point_positions, self.pad_length)
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions:Optional[torch.Tensor] = None):
        x= x[..., self.pad_length:-self.pad_length]
        if point_positions is not None:
            point_positions = point_positions[..., self.pad_length:-self.pad_length]
            return x, point_positions
        return x
    
class ReplicationPad(nn.Module):
    def __init__(self, pad_length):
        super().__init__()
        self.pad_length = pad_length

    def forward(self, x:torch.Tensor, point_positions:Optional[torch.Tensor] = None):
        assert x.shape[-1] // 2 >= self.pad_length, f"Input length is too short for padding: {x.shape[-1]}"
        x = torch.nn.functional.pad(x, (self.pad_length, self.pad_length), mode='replicate')
        if point_positions is not None:
            point_positions = pad_point_positions(point_positions, self.pad_length)
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions:Optional[torch.Tensor] = None):
        x= x[..., self.pad_length:-self.pad_length]
        if point_positions is not None:
            point_positions = point_positions[..., self.pad_length:-self.pad_length]
            return x, point_positions
        return x