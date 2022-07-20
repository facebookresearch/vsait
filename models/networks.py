# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.utils.spectral_norm as spectral_norm
from torch import nn
from torch.nn import functional as F


def get_pad_fn(pad_type="zero"):
    """get pad_fn based on pad_type in {"reflect","replicate","zero"}"""
    if pad_type.lower() == "reflect":
        pad_fn = nn.ReflectionPad2d
    elif pad_type.lower() == "replicate":
        pad_fn = nn.ReplicationPad2d
    elif pad_type.lower() in ["zero", None]:
        pad_fn = nn.ZeroPad2d
    else:
        raise Exception('unkonwn pad_type "%s"' % pad_type)
    return pad_fn


class ConvBlock(nn.Module):
    """
    Basic convolution block with conv, norm, activation

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        normalization: normalization function [default: torch.nn.InstanceNorm2d]
        activation: activation function [default: torch.nn.LeakyReLU(0.2)]
        deconv: True/False use torch.nn.ConvTranspose2d instead of torch.nn.Conv2d
            [default: False]
        kernel_size: kernel size for convolution [default: 3]
        padding: padding for convolution [default: 1]
        pad_type: padding type in ['reflect','replicate','zero',None]
            if `pad_type in ['zero',None]`, nn.ZeroPad2d is used. [default: 'zero']
        spectral_norm_conv: True/False use spectral norm convolution
        **kwargs: keyword arguments passed to torch.nn.Conv2d or torch.nn.ConvTranspose2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization=nn.InstanceNorm2d,
        activation=None,
        deconv=False,
        kernel_size=3,
        padding=1,
        pad_type="zero",
        spectral_norm_conv=False,
        **kwargs,
    ):
        super(ConvBlock, self).__init__()
        activation = nn.LeakyReLU(0.2) if activation is None else activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad = get_pad_fn(pad_type)(padding)

        # set conv
        if deconv:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, **kwargs
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

        # spectral norm conv layer
        if spectral_norm_conv:
            self.conv = spectral_norm(self.conv)

            # remove bias if noramlization
            if normalization and getattr(self.conv, "bias", None) is not None:
                delattr(self.conv, "bias")
                self.conv.register_parameter("bias", None)

        # set norm, activation
        self.norm = normalization(out_channels) if normalization else nn.Sequential()
        self.act = nn.Sequential() if activation is None else activation

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class LinearBlock(nn.Module):
    """
    Basic linear block with linear, norm, activation

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        normalization: normalization function [default: None]
        activation: activation function [default: torch.nn.ReLU()]
        dropout: dropout probability [default: 0., no dropout]

    Note:
        The `normalization` should be a function that is passed the number of output
        channels (i.e., `nn.InstanceNorm2d` not `nn.InstanceNorm2d(64)`).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        normalization=None,
        activation=None,
        dropout=0.0,
    ):
        super(LinearBlock, self).__init__()
        activation = nn.ReLU() if activation is None else activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = normalization(out_channels) if normalization else nn.Sequential()
        self.act = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.flatten(1)
        x = self.linear(x)
        x = self.norm(x)
        return self.dropout(self.act(x))


class MLP(nn.Sequential):
    """
    Multilayer perceptron with multiple LinearBlocks/ConvBlocks

    Args:
        in_channels: number of input channels
        n_channels: list of output channels for each LinearBlock in MLP
        normalization: normalization function for each layer except last, which is None
            [default: None]
        activation: activation function for each layer except last, which is None
            [default: torch.nn.ReLU()]
        dropout: dropout probability for each layer except last, which is 0.
            [default: 0., no dropout]
        conv: True/False use 1x1 convolutional layers, otherwise `nn.Linear`
            [default: False]
        **kwargs: keyword arguments passed `ConvBlock` if `conv is True`

    Note:
        The `normalization` should be a function that is passed the number of output
        channels (i.e., `nn.InstanceNorm2d` not `nn.InstanceNorm2d(64)`).
    """

    def __init__(
        self,
        in_channels,
        n_channels,
        normalization=None,
        activation=None,
        dropout=0.0,
        conv=False,
        **kwargs,
    ):
        activation = nn.ReLU() if activation is None else activation
        net = []
        n_layers = len(n_channels)
        for i, ch in enumerate(n_channels):
            # set in_ch, norm, activ
            in_ch = in_channels if i == 0 else n_channels[i - 1]
            norm = nn.Sequential() if i == (n_layers - 1) else normalization
            activ = nn.Sequential() if i == (n_layers - 1) else activation
            dropout = 0.0 if i == (n_layers - 1) else dropout

            # append conv/linear block
            if conv:
                kwargs.setdefault("kernel_size", 1)
                kwargs.setdefault("padding", 0)
                net.append(ConvBlock(in_ch, ch, norm, activ, **kwargs))
            else:
                net.append(LinearBlock(in_ch, ch, norm, activ))

        super(MLP, self).__init__(*net)


class ResNetBlock(nn.Module):
    """
    Basic ResNet Block

    Args:
        n_channels: number of channels in each convolutional layer
            (both input and output channels)
        normalization: normalization function [default: torch.nn.InstanceNorm2d]
        activation: activation function [default: torch.nn.LeakyReLU(0.2)]
        output_activation: activation function for output of second convolution
            [default: None]
        kernel_size: kernel size for convolution [default: 3]
        padding: padding for convolution [default: 1]
        pad_type: padding type choice of {'reflect','replicate','zero',None}
            if `pad_type in ['zero',None]`, nn.ZeroPad2d is used. [default: 'zero']
        **kwargs: keyword arguments passed to torch.nn.Conv2d

    Note:
        The `normalization` should be a function that is passed the number of output
        channels (i.e., `nn.InstanceNorm2d` not `nn.InstanceNorm2d(64)`).
    """

    def __init__(
        self,
        n_channels,
        normalization=nn.InstanceNorm2d,
        activation=None,
        output_activation=None,
        kernel_size=3,
        padding=1,
        pad_type="zero",
        **kwargs,
    ):
        super(ResNetBlock, self).__init__()
        activation = nn.LeakyReLU(0.2) if activation is None else activation
        self.n_channels = n_channels
        self.in_channels = self.out_channels = n_channels
        self.output_activation = (
            nn.Sequential() if output_activation is None else output_activation
        )

        # set conv_blocks
        conv_blocks = []
        conv_blocks.append(
            ConvBlock(
                n_channels,
                n_channels,
                normalization=normalization,
                activation=activation,
                kernel_size=kernel_size,
                padding=padding,
                pad_type=pad_type,
                **kwargs,
            )
        )
        conv_blocks.append(
            ConvBlock(
                n_channels,
                n_channels,
                normalization=normalization,
                activation=torch.nn.Sequential(),
                kernel_size=kernel_size,
                padding=padding,
                pad_type=pad_type,
                **kwargs,
            )
        )
        self.net = nn.Sequential(*conv_blocks)

    def forward(self, x):
        return self.output_activation(x + self.net(x))


class GaussianConv2d(nn.Module):
    """
    Gaussian up/downsampling

    Args:
        in_channels: number of input channels
        kernel_size: kernel size of gaussian kernel [default: 3]
        stride: stride of convolution [default : 2]
        padding: padding for convolution [default : 0]
        pad_type: padding type choice of {'reflect','replicate','zero',None}
            if `pad_type in ['zero',None]`, nn.ZeroPad2d is used. [default: 'zero']
        deconv: if True, conv_transpose2d is used (upsample), else conv2d (downsample)
            [default: False]
    """

    def __init__(
        self,
        in_channels,
        kernel_size=3,
        stride=2,
        padding=0,
        pad_type="zero",
        deconv=False,
    ):
        super(GaussianConv2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = deconv

        # set conv/conv_transpose
        kernel_pad = int((kernel_size - 1) / 2)
        if deconv:
            self.conv = F.conv_transpose2d
            self.padding = kernel_pad + 1
            padding += 1
        else:
            self.conv = F.conv2d
            self.padding = 0
            padding += kernel_pad + int(kernel_size % 2 == 0)

        # set slice for output
        start = int(deconv)
        end = None if kernel_size % 2 else -1
        self.slice = slice(start, end)

        # set padding
        self.pad = get_pad_fn(pad_type)(padding)

        # get gaussian weight
        weight = self.gaussian_kernel(kernel_size)
        if deconv:
            weight *= stride**2
        self.register_buffer("weight", weight.repeat(in_channels, 1, 1, 1))

    def gaussian_kernel(self, kernel_size):
        """2-d gaussian kernel with sigma=1"""
        xy = (
            torch.stack(torch.meshgrid((torch.arange(kernel_size),) * 2), dim=0)
            .unsqueeze(0)
            .float()
        )
        mu = torch.tensor(kernel_size / 2.0).view(-1, 1, 1) - 0.5
        g = torch.exp(-torch.sum((xy - mu) ** 2.0, dim=-3) / 2)
        return g / g.sum()

    def forward(self, x):
        out = self.conv(
            self.pad(x),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels,
        )
        return out[:, :, self.slice, self.slice]
