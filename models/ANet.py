import torch.nn as nn

from models.blocks import ResBlock, ConvDownsample

class ANet(nn.Module):
    def __init__(self, input_size: int = 256, latent_dim: int = 2048):
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
            ResBlock(128, 128),
            ConvDownsample(),
            ResBlock(128, 256),
            ResBlock(256, 256),
            ConvDownsample(),
            ResBlock(256, 256),
            ResBlock(256, 256),
            nn.Flatten(),
        )

        self.linear = nn.Linear(input_size / 16 * 256, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.res_blocks(x)
        return self.linear(x)