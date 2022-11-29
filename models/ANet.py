import torch
import torch.nn as nn

from models.blocks import ResBlock, ConvDownsample


class ANet(nn.Module):
    def __init__(self, input_size: int = 256, latent_dim: int = 2048):
        super().__init__()

        self.res_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 1024),
            ResBlock(1024, 1024),
            nn.Flatten(),
        )
        self.linear = nn.Linear((input_size // 32) ** 2 * 1024, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.res_blocks(x)
        return self.linear(x)
