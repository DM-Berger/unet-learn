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

from model.conv import ConvUnit


class EncodeBlock(nn.Module):
    """The downward / encoding layers of the U-Net"""

    def __init__(
        self, features_out: int, depth: int, normalization: bool = True, is_input: bool = False
    ):
        super().__init__()
        norm_first = not is_input
        inch, ouch = self._in_out_channels(depth, features_out)
        self.conv0 = ConvUnit(
            in_channels=inch[0], out_channels=ouch[0], normalization=norm_first, kernel_size=3
        )
        self.conv1 = ConvUnit(
            in_channels=inch[1], out_channels=ouch[2], normalization=normalization, kernel_size=3
        )
        self.pool = MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv0(x)
        x = self.conv1(x)
        skip = x
        x = self.pool(x)  # type: ignore
        return x, skip

    @staticmethod
    def _in_out_channels(
        depth: int, features_out: int = 32
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Abstract counting logic. Returns dicts in_ch, out_ch."""
        if depth == 0:  # input layer
            in0 = 1
            in1 = out0 = features_out
        else:
            in1 = out0 = in0 = features_out * 2 ** (depth)
        out1 = in1 * 2
        return {0: in0, 1: in1}, {0: out0, 1: out1}


class Encoder(nn.Module):
    def __init__(self, features_out: int = 32, depth: int = 3, normalization: bool = True):
        """Build the encoding side (downward convolution portion, left side of the U).

        Parameters
        ----------
        features_out: int
            The number of features to extract in the very first convolution
            layer (e.g. size of the very first filter bank in the first
            convolutional block).

        depth: int
            How deep the U-Net should be, not including the bottom layer (e.g.
            layer without skip connections). The classic Ronnberger 3D U-Net
            thus has depth 3.
        """
        super().__init__()
        self.depth_ = depth
        self.features_out = f = features_out
        self.blocks = nn.ModuleList()

        for d in range(depth):
            self.blocks.append(
                EncodingBlock(features_out=f, depth=d, normalization=True, is_input=(d == 0))
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        skips: List[Tensor] = []
        for i, encode in enumerate(self.blocks):
            x, skips[i] = encode(x)
        return x, skips
