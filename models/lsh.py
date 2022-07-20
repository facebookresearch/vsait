# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from . import utils


class LSH(torch.nn.Module):
    """
    Locality-Sensitive Hashing

    Project a feature vector or patch of feature vectors to a normalized
    space [-1,1] using a random weight matrix R containing unit-normalized rows

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size for convolution (input is unit normalized wrt kernel size)
        concatenate: True/False store sum of squares rather than normalize. Specifically
            used for `LSHCat` to concatenate outputs together. [default: False]
        **kwargs: keyword arguments passed to `torch.conv2d`

    Note:
        Input features are normalized with respect to the convolution kernel such that
        the output vectors are the cosine similarity with each row in the random matrix R.
        The weight matrix R is not learned, but this approach is useful for projecting
        features into a normalized space for vector symbolic architectures (VSAs).
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, concatenate=False, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = F._pair(kernel_size)
        self.concatenate = concatenate

        w = torch.randn(out_channels, in_channels, *self.kernel_size)
        if self.concatenate:
            self.register_buffer(
                "ssq_weight", torch.sum(w**2, dim=[1, 2, 3]).view(1, -1, 1, 1)
            )
            self.register_buffer("_ssq_conv", torch.ones(1, 1, *self.kernel_size))
        else:
            w = F.normalize(w, dim=[1, 2, 3])

        self.register_buffer("weight", w)

        self.conv_kwargs = kwargs

    def forward(self, x):
        """
        Apply LSH random projection

        Args:
            x: torch.Tensor input
        """
        # store ssq_x if concatenating
        if self.concatenate:
            self.ssq_x = torch.conv2d(
                torch.sum(x**2, dim=1, keepdim=True),
                self._ssq_conv,
                **self.conv_kwargs,
            )
        else:
            # normalize x
            h, w = x.shape[2:]
            x = F.unfold(x, self.kernel_size, **self.conv_kwargs)
            x = F.normalize(x, dim=1)
            x = F.fold(x, (h, w), self.kernel_size, **self.conv_kwargs)
        # convolve with normalized weight
        return torch.conv2d(x, self.weight, **self.conv_kwargs)


class LSHCat(torch.nn.ModuleList):
    """
    Locality-Sensitive Hashing with concatenated inputs

    Project a list of inputs by concatenating the input vectors for LSH

    Args:
        in_channels: list of number of input channels per input
        out_channels: number of output channels
        kernel_size: kernel size for convolution (input is unit normalized wrt kernel size)
            if list, kernel size per input
        **kwargs: keyword arguments passed to `torch.conv2d`. similar to `kernel_size`
            each kwarg value can also be a list of values per input

    Note:
        Separate LSH networks are initialized for each input but are applied as if
        inputs and projection weights were concatenated together for computing the
        cosine similarity (i.e., inputs and weights are normalized wrt all vectors).
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels

        kwargs.update(
            {
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "concatenate": True,
            }
        )

        for i, ch in enumerate(self.in_channels):
            self.append(LSH(ch, **utils.parse_dict_var(kwargs, i)))

    def _apply_lsh(self, fn, x):
        # update w_norm
        w_norm = getattr(self, "w_norm", 0.0)
        w_norm += fn.ssq_weight
        self.w_norm = w_norm

        # apply lsh, get ssq_x
        x = fn(x)
        x_norm = fn.ssq_x

        # get x_out, x_norm
        x_out = getattr(self, "x_out", x)
        x_norm = getattr(self, "x_norm", x_norm)

        # update x_out/x_norm with resizing
        if hasattr(self, "x_out") and hasattr(self, "x_norm"):
            x_out = x_out + F.interpolate(x, size=x_out.shape[2:], mode="nearest")
            x_norm = x_norm + F.interpolate(
                fn.ssq_x, size=x_out.shape[2:], mode="nearest"
            )

        # update x_out, x_norm
        self.x_out = x_out
        self.x_norm = x_norm

    def _compute_total_cosine_similarity(self):
        # assert variables exist
        assert (
            hasattr(self, "x_out")
            and hasattr(self, "x_norm")
            and hasattr(self, "w_norm")
        ), "Missing stored variables. `_apply_lsh` must be called first."

        # compute total cosine similarity, reset variables
        out = self.x_out / (self.x_norm.sqrt() * self.w_norm.sqrt() + 1e-5)
        delattr(self, "x_out")
        delattr(self, "x_norm")
        delattr(self, "w_norm")
        return out

    def forward(self, x_list):
        """
        Concatenate and project list of tensors

        Args:
            x_list: list of tensors to concatenate and project
        """
        # apply each lsh
        for i, fn in enumerate(self):
            x = x_list[i] if isinstance(x_list, list) else x_list
            self._apply_lsh(fn, x)

        # compute total cosine similarity
        return self._compute_total_cosine_similarity()
