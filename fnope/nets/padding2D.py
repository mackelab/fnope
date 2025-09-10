import torch
import torch.nn as nn
from typing import Optional
 
def get_padding(type = "zero", pad_length = 0):
    if pad_length == 0 or type == 'none':
        return IdentityPad2d()
    type = type.lower()
    if type == 'zero':
        return ZeroPad2d(pad_length)
    elif type == 'circular':
        return CircularPad2d(pad_length)
    elif type == 'reflect':
        return ReflectionPad2d(pad_length)
    else:
        raise ValueError(f"Invalid padding type: {type}")
    

#### CURRENT IMPLEMENTATION ONLY FOR FFT, SO POINT POSITIONS NOT NEEDED ####

class IdentityPad2d(nn.Module):
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

class ZeroPad2d(nn.Module):
    def __init__(self, pad_length):
        super().__init__()
        self.pad_length = pad_length

    def forward(self, x: torch.Tensor, point_positions: Optional[torch.Tensor] = None):

        batchsize = x.shape[0]
        in_channel = x.shape[1]
        domain_size_x = x.shape[2]
        domain_size_y = x.shape[3]
        
        assert domain_size_x // 2 >= self.pad_length and domain_size_y // 2 >= self.pad_length, f"Domain size {domain_size_x}x{domain_size_y} is too short for padding: {self.pad_length}"
        
        # Padding in last two dimensions
        x = torch.nn.functional.pad(x, (self.pad_length, self.pad_length, self.pad_length, self.pad_length), mode='constant', value=0)
        if point_positions is not None:
            #Currently point position padding doesnt work well
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions: Optional[torch.Tensor] = None):

        batchsize = x.shape[0]
        in_channel = x.shape[1]
        domain_size_x = x.shape[2]
        domain_size_y = x.shape[3]

        # Remove padded parts from data 
        x = x[:, :, self.pad_length:-self.pad_length, self.pad_length:-self.pad_length]
        if point_positions is not None:
            #Currently point position padding doesnt work well
            return x, point_positions
        return x

class CircularPad2d(nn.Module):
    
    def __init__(self, pad_length):
        super().__init__()
        self.pad_length = pad_length

    def forward(self, x: torch.Tensor, point_positions: Optional[torch.Tensor] = None):

        batchsize = x.shape[0]
        in_channel = x.shape[1]
        domain_size_x = x.shape[2]
        domain_size_y = x.shape[3]
        
        assert domain_size_x // 2 >= self.pad_length and domain_size_y // 2 >= self.pad_length, f"Domain size {domain_size_x}x{domain_size_y} is too short for padding: {self.pad_length}"
        
        # Padding in the last two dimensions
        x = torch.nn.functional.pad(x, (self.pad_length, self.pad_length, self.pad_length, self.pad_length), mode='circular')
        if point_positions is not None:
            #Currently point position padding doesnt work well
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions: Optional[torch.Tensor] = None):

        batchsize = x.shape[0]
        in_channel = x.shape[1]
        domain_size_x = x.shape[2]
        domain_size_y = x.shape[3]

        # Remove padded parts from data 
        x = x[:, :, self.pad_length:-self.pad_length, self.pad_length:-self.pad_length]
        if point_positions is not None:
            #Currently point position padding doesnt work well
            return x, point_positions
        return x

class ReflectionPad2d(nn.Module):
    def __init__(self, pad_length):
        super().__init__()
        self.pad_length = pad_length

    def forward(self, x: torch.Tensor, point_positions: Optional[torch.Tensor] = None):

        batchsize = x.shape[0]
        in_channel = x.shape[1]
        domain_size_x = x.shape[2]
        domain_size_y = x.shape[3]
        
        assert domain_size_x // 2 >= self.pad_length and domain_size_y // 2 >= self.pad_length, f"Domain size {domain_size_x}x{domain_size_y} is too short for padding: {self.pad_length}"

        # Padding in the last two dimensions
        x = torch.nn.functional.pad(x, (self.pad_length, self.pad_length, self.pad_length, self.pad_length), mode='reflect')
        if point_positions is not None:
            #Currently point position padding doesnt work well
            return x, point_positions
        return x
    
    def unpad(self, x, point_positions: Optional[torch.Tensor] = None):
        
        batchsize = x.shape[0]
        in_channel = x.shape[1]
        domain_size_x = x.shape[2]
        domain_size_y = x.shape[3]

        # Remove padded parts from data 
        x = x[:, :, self.pad_length:-self.pad_length, self.pad_length:-self.pad_length]
        if point_positions is not None:
            #Currently point position padding doesnt work well
            return x, point_positions
        return x