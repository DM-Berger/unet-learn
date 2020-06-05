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


"""
NOTEs

You might want to
"""


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


class EncodingBlock(nn.Module):
    """The downward / encoding layers of the U-Net"""

    def __init__(
        self,
        features_out: int,
        depth: int,
        normalization: bool = True,
        pool: bool = True,
        is_input: bool = False,
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
        self.pool = MaxPool3d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.conv0(x)
        x = self.conv1(x)
        if not self.pool:  # really just for last / bottom block
            return x, None
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
            How deep the U-Net should be, including the bottom layer (e.g.
            layer without skip connections). The classic Ronnberger 3D U-Net
            thus has depth 4.
        """
        super().__init__()
        self.depth_ = depth
        self.features_out = features_out
        self.blocks = nn.ModuleList()

        for d in range(depth):
            self.blocks.append(
                EncodingBlock(
                    features_out=features_out,
                    depth=d,
                    normalization=True,
                    pool=(d == (depth - 1)), # don't downsample on bottom
                    is_input=(d == 0),
                )
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Optional[Tensor]]]:
        skips: List[Optional[Tensor]] = []
        for i, encode in enumerate(self.blocks):
            x, skips[i] = encode(x)
        return x, skips

class DecodeBlock(nn.Module):
    """The upsampling layer, followed by two ConvUnits, the first of which takes
    the skip connection.

    Note that almost all the params of the decoder are dependent on those chosen for
    the Encoder. We can use this to our advantage in construction."""
    def __init__(self, encoder: Encoder, depth: int, normalization: bool = True):
        super().__init__()
        inch, ouch, upch = self._in_out_channels(depth, encoder.features_out)
        self.upconv = UpConv3d(in_channels=upch, out_channels=upch, kernel_size=2, stride=2)
        self.conv0 = ConvUnit(in_channels=inch[0], out_channels=ouch[0], normalization=normalization)
        self.conv1 = ConvUnit(in_channels=inch[1], out_channels=ouch[1], normalization=normalization)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upconv(x)
        # remember, batch tensors are shape [B_size, Ch_size, <InputDims>]
        x = torch.cat([x, skip], dim=1)  # concat along channel dimension
        x = self.conv0(x)
        x = self.conv1(x)
        return x

    @staticmethod
    def _in_out_channels(depth: int, features_out: int = 32) -> Tuple[Dict[int, int], Dict[int, int], int]:
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
        skip = f*2**(depth+1)  # number of in_channels / features of the skip connection
        in0 = 3*skip  # first in is always skip + 2*skip channels
        out1 = in1 = out0 = skip
        up = up_out = up_in = 2*skip
        return {0: in0, 1: in1}, {0: out0, 1: out1}, up

class Decoder(nn.Module):
    def __init__(self, encoder: Encoder, normalization: bool = True):
        super().__init__()
        self.depth_ = encoder.depth_
        self.features_out = encoder.features_out
        self.blocks = nn.ModuleList()

        for depth in range(self.depth):
            self.blocks.append(DecodeBlock(encoder, depth, normalization))

    def forward(self, x, skip)


def DoubleConv3D(
    in_ch: int, out_ch: int, down: bool = True, batch_norm: bool = True, label: str = None
) -> Sequential:
    if label is None:
        label = f"DownConvBlock3d-{in_ch}in" if down else f"UpConvBlock3d-{in_ch}in"

    conv1 = (
        f"{label}.conv1",
        Conv3d(in_channels=in_ch, out_channels=out1, kernel_size=3, stride=1),
    )
    batch1 = (f"{label}.batch1", BatchNorm3d(num_features=out1, track_running_stats=True))
    relu1 = (f"{label}.relu1", ReLU(inplace=True))
    conv2 = (f"{label}.conv2", Conv3d(in_channels=in2, out_channels=out2, kernel_size=3, stride=1))
    batch2 = (f"{label}.batch2", BatchNorm3d(num_features=out2, track_running_stats=True))
    relu2 = (f"{label}.relu2", ReLU(inplace=True))
    # not we can't include the max-pooling layer because we need access to the
    # output for our skip connections via torch.cat
    if batch_norm:
        return nn.Sequential(OrderedDict([conv1, batch1, relu1, conv2, batch2, relu2]))
    else:
        return nn.Sequential(OrderedDict([conv1, relu1, conv2, relu2]))


# Implementation of the 3D Ronnberger U-Net (arXiv: 1606.06650v1)
# for Biomedical Image Segmentation
# The code below draws from:
# https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
# https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
def DownConvBlock3D(
    i: int, initial_out: int = 32, batch_norm: bool = True, label: str = None
) -> Sequential:
    """Build a U-Net 'block' consisting of two 3x3x3 Convolution layers each
    followed by ReLU activations.

    Parameters
    ----------
    i: int
        Denotes the block depth. I.e. for i == 0, this is the input block. If
        building the classic Ronnberger 3D U-Net, then the maximum value of i
        is i == 3. That is:
          - i == 0 corresponds to the:    3 > 32  > 64   input layer (top)
          - i == 1 corresponds to the:   64 > 64  > 128  layer
          - i == 2 corresponds to the:  128 > 128 > 256  layer
          - i == 3 corresponds to the:  256 > 256 > 512  layer (bottom)

        For the first block (i==0), `in_channels` should be equal to `1`, since the
        input image is an MRI, and is only black and white.

        If any other block, the number of input channels should be equal to the
        number of output channels (i.e. 'learned features', or number of
        filters) of the previous DownConvBlock3D.

    initial_out: int
        The size of the first flter bank / number of learned features.

    batch_norm: bool
        Whether or not to apply batch normalization prior to each ReLU
        activation.

    label: Optional[str]
        An identifier for the block. If None, will be f"DownConvBlock3d-{i}"

    Returns
    -------
    model: torch.nn.Sequential

    Notes
    -----
    The "classic" U-Net architecture of Ronnberger et all is quite simple and
    elegant, and has an implicit maximum `U-depth` defined by the number of
    initial filters / features in the first convolution layer. This is true
    regardless of the input size (so long as batch size is 1, unless using a
    custom trainer outside of the usual Torch / Theano / Tensorflow tools),
    as the U-Net is fully convolutional (i.e. there are no dense /
    fully-connected layers that require sizing information).
    """

    in1 = 1 if i == 0 else initial_out * (2 ** (i - 1))
    in2 = initial_out if i == 0 else in1 * 2
    out1 = initial_out if i == 0 else in1 * 2
    out2 = out1 * 2
    label = f"DownConvBlock3d-{i}" if label is None else label

    conv1 = (
        f"{label}.conv1.down",
        Conv3d(in_channels=in1, out_channels=out1, kernel_size=3, stride=1),
    )
    batch1 = (f"{label}.batch1.down", BatchNorm3d(num_features=out1, track_running_stats=True))
    relu1 = (f"{label}.relu1", ReLU(inplace=True))
    conv2 = (
        f"{label}.conv2.down",
        Conv3d(in_channels=in2, out_channels=out2, kernel_size=3, stride=1),
    )
    batch2 = (f"{label}.batch2.down", BatchNorm3d(num_features=out2, track_running_stats=True))
    relu2 = (f"{label}.relu2", ReLU(inplace=True))
    # not we can't include the max-pooling layer because we need access to the
    # output for our skip connections via torch.cat
    if batch_norm:
        return torch.nn.Sequential(OrderedDict([conv1, batch1, relu1, conv2, batch2, relu2]))
    else:
        return torch.nn.Sequential(OrderedDict([conv1, relu1, conv2, relu2]))


def UpConvBlock3D(n_features: int, label: str, pool_stride: int = 1) -> Sequential:
    out = n_features
    upconv = lambda i: (
        f"{label}.conv{i}.up",
        UpConv3d(in_channels=1, out_channels=out, kernel_size=2, stride=1),
    )
    return torch.nn.Sequential(OrderedDict([upconv(1), upconv(2)]))


def UpConvBlock3D(i: int, initial_out: int = 32, label: str = None) -> Sequential:
    """Build a U-Net 'block' consisting of two 3x3x3 Convolution layers each
    followed by ReLU activations.

    Parameters
    ----------
    i: int
        Denotes the block depth. I.e. for i == 0, this is top block of
        convolutions in the contracting layer. For Ronnberger's 3D U-net, the
        the maximum value of is i == 3. That is:
          - i == 0 corresponds to the:   64 + 128 >  64 >  64  output block (top)
          - i == 1 corresponds to the:  128 + 256 > 128 > ___  layer
          - i == 2 corresponds to the:  256 + 512 > 128 > 256  layer
          - i == 3 corresponds to the:  512 > 256 > 512  layer (bottom)

        For the first block (i==0), `in_channels` should be equal to `1`, since the
        input image is an MRI, and is only black and white.

        If any other block, the number of input channels should be equal to the
        number of output channels (i.e. 'learned features', or number of
        filters) of the previous DownConvBlock3D.

    Returns
    -------
    model: torch.nn.Sequential

    Notes
    -----
    The "classic" U-Net architecture of Ronnberger et all is quite simple and
    elegant, and has an implicit maximum `U-depth` defined by the number of
    initial filters / features in the first convolution layer. This is true
    regardless of the input size (so long as batch size is 1, unless using a
    custom trainer outside of the usual Torch / Theano / Tensorflow tools),
    as the U-Net is fully convolutional (i.e. there are no dense /
    fully-connected layers that require sizing information).
    """

    in1 = 1 if i == 0 else initial_out * (2 ** (i - 1))
    in2 = initial_out if i == 0 else in1 * 2
    out1 = initial_out if i == 0 else in1 * 2
    out2 = out1 * 2
    label = f"UpConvBlock3d-{i}" if label is None else label

    upconv = (
        f"{label}.upconv.up",
        UpConv3d(in_channels=in1, out_channels=out1, kernel_size=2, stride=2),
    )
    conv1 = (
        f"{label}.conv1.up",
        Conv3d(in_channels=in2, out_channels=out2, kernel_size=3, stride=1),
    )
    relu1 = (f"{label}.relu1.up", ReLU(inplace=True))
    conv2 = (
        f"{label}.conv2.up",
        Conv3d(in_channels=in3, out_channels=out3, kernel_size=3, stride=1),
    )
    relu2 = (f"{label}.relu2.up", ReLU(inplace=True))
    # not we can't include the max-pooling layer because we need access to the
    # output for our skip connections via torch.cat
    return torch.nn.Sequential(OrderedDict([upconv, conv1, relu1, conv2, relu2]))


def DownConvBlock3D(i: int, initial_out: int = 32, label: str = None) -> Sequential:
    """Build a U-Net 'block' consisting of two 3x3x3 Convolution layers each
    followed by ReLU activations.

    Parameters
    ----------
    i: int
        Denotes the block depth. I.e. for i == 0, this is the input block. If
        building the classic Ronnberger 3D U-Net, then the maximum value of i
        is i == 3. That is:
          - i == 0 corresponds to the:    3 > 32  > 64   input layer (top)
          - i == 1 corresponds to the:   64 > 64  > 128  layer
          - i == 2 corresponds to the:  128 > 128 > 256  layer
          - i == 3 corresponds to the:  256 > 256 > 512  layer (bottom)

        For the first block (i==0), `in_channels` should be equal to `1`, since the
        input image is an MRI, and is only black and white.

        If any other block, the number of input channels should be equal to the
        number of output channels (i.e. 'learned features', or number of
        filters) of the previous DownConvBlock3D.

    Returns
    -------
    model: torch.nn.Sequential

    Notes
    -----
    The "classic" U-Net architecture of Ronnberger et all is quite simple and
    elegant, and has an implicit maximum `U-depth` defined by the number of
    initial filters / features in the first convolution layer. This is true
    regardless of the input size (so long as batch size is 1, unless using a
    custom trainer outside of the usual Torch / Theano / Tensorflow tools),
    as the U-Net is fully convolutional (i.e. there are no dense /
    fully-connected layers that require sizing information).
    """

    in1 = 1 if i == 0 else initial_out * (2 ** (i - 1))
    in2 = initial_out if i == 0 else in1 * 2
    out1 = initial_out if i == 0 else in1 * 2
    out2 = out1 * 2
    label = f"DownConvBlock3d-{i}" if label is None else label

    conv1 = (
        f"{label}.conv1.down",
        Conv3d(in_channels=in1, out_channels=out1, kernel_size=3, stride=1),
    )
    batch1 = (f"{label}.batch1.down", BatchNorm3d(num_features=out1, track_running_stats=True))
    relu1 = (f"{label}.relu1", ReLU(inplace=True))
    conv2 = (
        f"{label}.conv2.down",
        Conv3d(in_channels=in2, out_channels=out2, kernel_size=3, stride=1),
    )
    batch2 = (f"{label}.batch2.down", BatchNorm3d(num_features=out2, track_running_stats=True))
    relu2 = (f"{label}.relu2", ReLU(inplace=True))
    # not we can't include the max-pooling layer because we need access to the
    # output for our skip connections via torch.cat
    return torch.nn.Sequential(OrderedDict([conv1, batch1, relu1, conv2, batch2, relu2]))
