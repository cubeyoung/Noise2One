import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .models import register, make
from utils import make_coord
from models.networks import UNet_Blind
import numpy as np
from einops import rearrange
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

@register('light')
class Light(nn.Module):
    def __init__(self, imnet_spec=None,hidden_dim=64):
        super().__init__()
        hidden_dim = imnet_spec['args']['hidden_list'][0]
        self.in_dim = 3
        if imnet_spec is not None:
            self.imnet = make(imnet_spec, args={'in_dim':9})
        else:
            self.imnet = None       
        
        self.adapter = nn.Sequential(
            nn.Conv2d(self.in_dim, 128, 3, stride=1, padding=1),
            Sine(),
            nn.Conv2d(128, self.in_dim, 3, stride=1, padding=1),
            Sine())
        
        print(self.imnet)
        print(self.adapter)
        linear_params = sum(p.numel() for p in self.imnet.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())

        print('[Linear decoder] Total number of parameters : %.3f M' % (linear_params / 1e6))
        print('[Noise Adapter] Total number of parameters : %.3f M' % (adapter_params / 1e6))

    def query_rgb(self, input, score, noise_level=None):
        b, c, h, w = input.size()
        # The line `weights = self.adapter(input)` is passing the `input` tensor through the `adapter`
        # module defined in the `Light` class. The `adapter` module consists of a series of operations
        # including convolutional layers and the `Sine` activation function.
        # weights = self.adapter(input)
        weights = torch.ones_like(input)*noise_level
        input = torch.cat([input,score,weights],dim=1)       
        input = rearrange(input, 'b c h w -> (b h w) c')     
        pred = self.imnet(input) 
        pred = rearrange(pred, '(b h w) c -> b c h w', b=b, h=h, w=w)  # Reshape back to image)     

        return pred 

    def forward(self, inp, score, noise_level):
        return self.query_rgb(inp, score, noise_level)