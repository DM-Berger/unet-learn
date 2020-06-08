import torch.nn as nn

from torch import Tensor
from torch.nn import Conv3d

from model.conv import ConvUnit
from model.encode import Encoder
from model.decode import Decoder


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
        normalization: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(initial_features, depth, normalization)
        self.joiner = JoinBlock(self.encoder, normalization)
        self.decoder = Decoder(self.encoder, normalization)
        self.segmenter = Conv3d(
            in_channels=2 * initial_features, out_channels=n_labels, kernel_size=1, stride=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x, skips = self.encoder(x)
        x = self.joiner(x)
        x = self.decoder(x, skips)
        x = self.segmenter(x)
        return x
