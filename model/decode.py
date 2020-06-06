import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import ConvTranspose3d as UpConv3d
from typing import Dict, List, Tuple

from model.conv import ConvUnit
from model.encode import Encoder


class DecodeBlock(nn.Module):
    """The upsampling layer, followed by two ConvUnits, the first of which takes
    the skip connection.

    Note that almost all the params of the decoder are dependent on those chosen for
    the Encoder. We can use this to our advantage in construction."""

    def __init__(self, encoder: Encoder, depth: int, normalization: bool = True):
        super().__init__()
        inch, ouch, upch = self._in_out_channels(depth, encoder.features_out)
        self.upconv = UpConv3d(in_channels=upch, out_channels=upch, kernel_size=2, stride=2)
        self.conv0 = ConvUnit(
            in_channels=inch[0], out_channels=ouch[0], normalization=normalization
        )
        self.conv1 = ConvUnit(
            in_channels=inch[1], out_channels=ouch[1], normalization=normalization
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upconv(x)
        # remember, batch tensors are shape [B_size, Ch_size, <InputDims>]
        x = torch.cat([x, skip], dim=1)  # concat along channel dimension
        x = self.conv0(x)
        x = self.conv1(x)
        return x

    @staticmethod
    def _in_out_channels(
        depth: int, features_out: int = 32
    ) -> Tuple[Dict[int, int], Dict[int, int], int]:
        """Abstract counting logic. Returns dicts in_ch, out_ch. Assumes an incoming skip connection.

        Parameters
        ----------
        depth: int
            depth == 0 is the last decoding block (e.g. top right of U-Net)

        features_out: int
            The amount of initial ouput features in the first convolution of the
            encoder.
        """
        f = features_out
        skip = f * 2 ** (depth + 1)  # number of in_channels / features of the skip connection
        in0 = 3 * skip  # first in is always skip + 2*skip channels
        out1 = in1 = out0 = skip
        up = 2 * skip
        return {0: in0, 1: in1}, {0: out0, 1: out1}, up


class Decoder(nn.Module):
    def __init__(self, encoder: Encoder, normalization: bool = True):
        super().__init__()
        self.depth_ = encoder.depth_
        self.features_out = encoder.features_out

        self.blocks = nn.ModuleList()
        for depth in range(self.depth_):
            self.blocks.append(DecodeBlock(encoder, depth, normalization))

    def forward(self, x: Tensor, skips: List[Tensor]) -> Tensor:
        for skip, decode in zip(skips, self.blocks):
            x = decode(x, skip)
        return x
