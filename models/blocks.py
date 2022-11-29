from typing import Type, Optional

import torch
import torch.nn as nn


class ConvDownsample(nn.Module):
    def __init__(self,
                 channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 act_type: Type = nn.ELU,
                 norm_type: Optional[Type] = nn.BatchNorm2d):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            act_type(),
            norm_type(out_channels) if norm_type is not None else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            act_type(),
            norm_type(out_channels) if norm_type is not None else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        return x + self.block(x)
 