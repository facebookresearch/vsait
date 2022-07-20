# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torch.nn import init

from . import utils
from .networks import ConvBlock


class Discriminator(nn.Module):
    """Base class for discriminators"""

    def __init__(self):
        super(Discriminator, self).__init__()

    def _set_init_defaults(self, init_type, kwargs):
        # set defaults for different init_types
        defaults = {
            "normal_": {"mean": 0.0, "std": 0.2},
            "xavier_normal_": {"gain": 0.2},
            "xavier_uniform_": {"gain": 1.0},
            "kaiming_normal_": {"a": 0.0, "mode": "fan_in"},
            "orthogonal_": {"gain": 0.2},
        }
        for k, v in defaults.get(init_type, {}).items():
            kwargs.setdefault(k, v)
        return kwargs

    def init_weights(self, init_type="xavier_normal_", **kwargs):
        # get defaults based on init_type
        kwargs = self._set_init_defaults(init_type, kwargs)
        # define initializer function
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, kwargs.get("std", 0.2))
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                classname.startswith("Conv") or classname.startswith("Linear")
            ):
                if init_type.lower() == "default":  # uses pytorch's default init method
                    m.reset_parameters()
                elif hasattr(init, init_type):
                    getattr(init, init_type)(m.weight.data, **kwargs)
                else:
                    raise NotImplementedError(
                        f"initialization method `{init_type}` is not implemented"
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, **kwargs)

    def build_net(self):
        pass

    def forward(self):
        pass


class PatchGANDiscriminator(Discriminator):
    """
    Convolutional discriminator network

    Args:
        in_channels: number of channels in input layer
        n_channels: channels per layer [default: [64, 128, 256, 512, 1]]
        kernel_size: kernel size used for convolutions [default: 4]
        stride: stride used for convolutions [default: 2]
        padding: padding used for convolutions [default: 1]
        normalization: normalization used for each layer (except first, which is None)
            [default: torch.nn.InstanceNorm2d]
        activation: activation function used for each layer (except last)
            [default: torch.nn.LeakyReLU(0.2)]
        last_activation: activation function for final layer [default: None]
        init_kwargs: kwargs passed to `init_weights` function
        **kwargs: additional attributes passed to `networks.ConvBlock`
    """

    def __init__(
        self,
        in_channels,
        n_channels=None,
        kernel_size=4,
        stride=2,
        padding=1,
        normalization=nn.InstanceNorm2d,
        activation=None,
        last_activation=None,
        init_kwargs=None,
        **kwargs,
    ):
        super(PatchGANDiscriminator, self).__init__()
        self.in_channels = in_channels
        n_channels = n_channels or [64, 128, 256, 512, 1]
        self.n_layers = len(n_channels)
        kernel_size = utils.parse_list_var(kernel_size, self.n_layers)
        stride = utils.parse_list_var(stride, self.n_layers)
        padding = utils.parse_list_var(padding, self.n_layers)

        self.net = self.build_net(
            in_channels,
            n_channels,
            normalization,
            nn.LeakyReLU(0.2) if activation is None else activation,
            last_activation,
            kernel_size,
            stride,
            padding,
            **kwargs,
        )
        self.init_weights(**(init_kwargs or {}))

    def build_net(
        self,
        in_channels,
        n_channels,
        normalization,
        activation,
        last_activation,
        kernel_size,
        stride,
        padding,
        **kwargs,
    ):
        """
        Build basic PatchGAN discriminator

        Args:
            in_channels: number of channels in input layer
            n_channels: channels per layer
            normalization: normalization used for each layer (except first, which is None)
            activation: activation function used for each layer (except last)
            last_activation: activation function for final layer
            kernel_size: kernel size used for convolutions
            stride: stride used for convolutions
            padding: padding used for convolutions
            **kwargs: additional attributes passed to `networks.ConvBlock`
        """
        assert (
            len(n_channels) > 2
        ), f"must have at least 3 layers, but found: {len(n_channels)}"
        # if normalization or activations str, eval from torch
        normalization = utils.parse_str_var(normalization)
        activation = utils.parse_str_var(activation)
        last_activation = utils.parse_str_var(last_activation)

        # if spectral norm, only apply to layers[1:-1]
        spectral_norm_conv = kwargs.pop("spectral_norm_conv", None)

        # set conv blocks
        conv_blocks = []
        conv_blocks.append(
            ConvBlock(
                in_channels,
                n_channels[0],
                normalization=None,
                activation=activation,
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
                spectral_norm_conv=False,
                **utils.parse_dict_var(kwargs, 0),
            )
        )
        for i in range(1, len(n_channels) - 1):
            conv_blocks.append(
                ConvBlock(
                    n_channels[i - 1],
                    n_channels[i],
                    normalization=normalization,
                    activation=activation,
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=padding[i],
                    spectral_norm_conv=spectral_norm_conv,
                    **utils.parse_dict_var(kwargs, i),
                )
            )
        conv_blocks.append(
            nn.Conv2d(
                n_channels[-2],
                n_channels[-1],
                kernel_size[-1],
                stride[-1],
                padding[-1],
            )
        )
        if last_activation is not None:
            conv_blocks.append(last_activation)

        return nn.Sequential(*conv_blocks)

    def forward(self, x):
        return self.net(x)


class VSADiscriminator(Discriminator):
    """
    Input image is passed through a pre-trained network, and extracted features
    from each layer are passed to a Locality-Sensitive Hashing (LSH) network
    and then to the discriminator.

    Args:
        in_channels: number of channels in input layer
        n_channels: number of channels per discriminator [default: [1024, 1024, 1]]
        kernel_size: kernel size used for convolutions [default: 1]
        stride: stride used for convolutions [default: 1]
        padding: padding used for convolutions [default: 0]
        normalization: normalization used for each layer (except first, which is None)
            [default: None]
        activation: activation function used for each layer (except last)
            [default: torch.nn.LeakyReLU(0.2)]
        num_scales: number of scales to pass image to discriminator via downsampling
            [default: 1, no downsampling]
        **kwargs: additional attributes passed to `PatchGANDiscriminator`
    """

    def __init__(
        self,
        in_channels,
        n_channels=None,
        kernel_size=1,
        stride=1,
        padding=0,
        normalization=None,
        activation=None,
        num_scales=1,
        **kwargs,
    ):
        super(VSADiscriminator, self).__init__()
        self.in_channels = in_channels
        n_channels = n_channels or [1024, 1024, 1]
        self.n_layers = len(n_channels)

        self.num_scales = num_scales
        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=False
        )

        self.net = self.build_net(
            in_channels,
            n_channels,
            normalization,
            activation,
            kernel_size,
            stride,
            padding,
            num_scales,
            **kwargs,
        )

    def build_net(
        self,
        in_channels,
        n_channels,
        normalization,
        activation,
        kernel_size,
        stride,
        padding,
        num_scales,
        **kwargs,
    ):
        """
        Build multi-scale PatchGAN discriminator

        Args:
            in_channels: number of channels in input layer
            n_channels: channels per layer
            normalization: normalization used for each layer (except first, which is None)
            activation: activation function used for each layer (except last)
            kernel_size: kernel size used for convolutions
            stride: stride used for convolutions
            padding: padding used for convolutions
            num_scales: number of scales to pass image to discriminator via downsampling
            **kwargs: additional attributes passed to `PatchGANDiscriminator`
        """
        return nn.ModuleList(
            [
                PatchGANDiscriminator(
                    in_channels,
                    n_channels=n_channels,
                    normalization=normalization,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=nn.LeakyReLU(0.2) if activation is None else activation,
                    **kwargs,
                )
                for _ in range(num_scales)
            ]
        )

    def forward(
        self,
        x,
        src=None,
        feature_extractor=None,
        lsh_feat=None,
        map_net=None,
    ):
        """
        Discriminate hypervectors extracted from pre-trained network.

        Args:
            x: torch.Tensor input image
            src: torch.Tensor corresponding source image if `x` is translated image
            feature_extractor: pre-trained feature extractor network
            lsh_feat: LSH network for random projection of extracted features
            map_net: hypervector mapping network (if `src is not None`)
        """
        out = []
        # pass features at each scale
        for i in range(self.num_scales):
            # get extracted features
            f = utils.make_list(feature_extractor(x, "target"))

            # apply LSH
            f = lsh_feat(f)

            # get discr outputs
            out.append(self.net[i](f))

            # get src and src_to_tgt hypervectors
            if src is not None:
                f_src = utils.make_list(feature_extractor(src, "source"))
                f_src = lsh_feat(f_src)
                src_to_tgt = map_net(f_src)

                out.append(self.net[i](f_src * src_to_tgt))

            # downsample x
            if self.num_scales > 1:
                x = self.downsample(x)
                if src is not None:
                    src = self.downsample(src)

        return out
