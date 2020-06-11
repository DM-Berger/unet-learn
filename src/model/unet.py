import torch.nn as nn

from torch import Tensor
from torch.nn import Conv3d, Sigmoid

from model.conv import ConvUnit
from model.encode import Encoder
from model.decode import Decoder


"""
NOTEs

You might want to
"""


class JoinBlock(nn.Module):
    """The bottom block of the U-Net, which performs no real down- or up-sampling

    Parameters
    ----------
    encoder: Encoder
        The Encoder module to be joined to a Decoder module.

    kernel_size: int
        The size of the kernels in the internal double convolution blocks /
        units.

    normalization: bool
        If True (default) whether or not to apply GroupNormalization3D in the
        convolutional units.
    """

    def __init__(self, encoder: Encoder, kernel_size: int = 3, normalization: bool = True):
        super().__init__()
        d, f = encoder.depth_, encoder.features_out
        in1 = out0 = in0 = f * (2 ** d)
        out1 = 2 * in1
        self.conv0 = ConvUnit(in0, out0, normalization, kernel_size=kernel_size)
        self.conv1 = ConvUnit(in1, out1, normalization, kernel_size=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        return x


class UNet3d(nn.Module):
    """Build the classic Ronnberger 3D U-Net.

    Parameters
    ----------
    initial_features: int
        The number of channels / filters / features to extract in the very first
        convolution. Sets the resulting sizes for the entire net.

    depth: int
        Encoder / decoder depth. E.g. a depth of 3 will results in 3 encoding
        blocks (two convolution layers + downsample), one joining bottom later, and 3
        decoding blocks (upsample + two convolution layers).

    normalization: bool
        If True, apply normalization (GroupNorm3D) after each non-upsampling or
        non-downsampling convolution within an encoding or decoding block.
    """

    def __init__(
        self,
        initial_features: int = 32,
        n_labels: int = 2,
        depth: int = 3,
        kernel_size: int = 3,
        normalization: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(initial_features, depth, kernel_size=kernel_size, normalization=normalization)
        self.joiner = JoinBlock(self.encoder, kernel_size=kernel_size, normalization=normalization)
        self.decoder = Decoder(self.encoder, kernel_size=kernel_size, normalization=normalization)
        self.segmenter = Conv3d(in_channels=2 * initial_features, out_channels=n_labels, kernel_size=1, stride=1)
        # use torch.nn.BCEWithLogitsLoss instead of sigmoid activation here

    def forward(self, x: Tensor) -> Tensor:
        x, skips = self.encoder(x)
        x = self.joiner(x)
        x = self.decoder(x, skips)
        x = self.segmenter(x)
        return x
