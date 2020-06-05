import torch.nn as nn

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

from model.conv import ConvUnit
from model.encode import EncodeBlock, Encoder
from model.decode import DecodeBlock, Decoder


"""
NOTEs

You might want to
"""


class JoinBlock(nn.Module):
    """The bottom block of the U-Net, which performs no real down- or up-sampling"""

    def __init__(self, encoder: Encoder, normalization: bool = True):
        super().__init__()
        d, f = encoder.depth_, encoder.features_out
        in1 = out0 = in0 = f * (2 ** d)
        out1 = 2 * in1
        self.conv0 = ConvUnit(in0, out0, normalization, kernel_size=3)
        self.conv1 = ConvUnit(in1, out1, normalization, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        return x
