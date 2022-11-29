import torch.nn as nn

from models.blocks import ResBlock, ConvDownsample

class PNet(nn.Module):
    def __init__(self, input_size: int = 256, output_size: int = 16, output_channels: int = 512):
        super().__init__()

        self.res_blocks = nn.Sequential(
            ResBlock(3, 8),
            ResBlock(8, 16),
            ResBlock(8, 32),
            ResBlock(32, 32),
            ConvDownsample(32),
            ResBlock(32, 64),
            ResBlock(64, 64),
            ConvDownsample(64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
            ConvDownsample(),
            ResBlock(256, 256),
            ResBlock(256, 512),
            ConvDownsample(),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )

    def forward(self, x: torch.Tensor):
        return self.res_blocks(x)