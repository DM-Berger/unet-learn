import torch
import torch.nn as nn

from collections import OrderedDict
from torch import Tensor
from torch.nn import (
    BatchNorm3d,
    Conv3d,
    ConvTranspose3d as UpConv3d,
    GroupNorm3d,
    MaxPool3d,
    ReLU,
    Sequential,
)
from typing import Dict, List, Optional, Tuple, Union

# just a clone of https://github.com/fepegar/unet/blob/v0.7.5/unet/conv.py for now.
class ConvUnit(nn.Module):
    """A combination convolution, (group) normalization, and activation layer"""

    def __init__(
        self, in_channels: int, out_channels: int, normalization: bool = True, kernel_size: int = 3
    ):
        super().__init__()
        block = nn.ModuleList()
        conv = Conv3d(in_channels, out_channels, kernel_size, padding=0)
        gnorm = GroupNorm3d(out_channels)
        act = ReLU(inplace=True)
        block.extend([conv, gnorm, act])
        self.block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        # tensors just flow straight through these
        return self.block(x)
