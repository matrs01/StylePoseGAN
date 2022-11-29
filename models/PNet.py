import torch
import torch.nn as nn

from models.blocks import ResBlock


class PNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.res_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(16),
            ResBlock(16, 16),
            ResBlock(16, 32),
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
            ResBlock(512, 512),
            ResBlock(512, 512),
        )

    def forward(self, x: torch.Tensor):
        return self.res_blocks(x)
