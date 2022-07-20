# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Compose(transforms.Compose):
    """
    Create transforms.Compose for image and non-image inputs performed
    together while applying appropriate augmentations based on non-image type.

    Args: fns (list of Transform objects): list of transforms to compose.
          additional_targets (dict): (key, value) mapping of input name and type
            where type is one of ['mask']

    Note: This is based conceptually off of `albumentations.Compose`
    """

    def __init__(self, fns, additional_targets=None):
        super().__init__(fns)
        self.additional_targets = additional_targets or {}
        self.ignore_fns = {
            "mask": [
                "Normalize",
            ]
        }

    def _call_fn_given_type(self, fn, k, v):
        t = self.additional_targets.get(k)
        # if fn has {type}_fn, apply
        if hasattr(fn, f"{t}_fn"):
            return getattr(fn, f"{t}_fn")(v)
        # if fn name in ignore_fns for type, return v
        elif fn.__class__.__name__ in self.ignore_fns.get(t, []):
            return v
        # apply fn
        return fn(v)

    def __call__(self, **kwargs):
        out = {}
        for k, v in kwargs.items():
            for fn in self.transforms:
                v = self._call_fn_given_type(fn, k, v)
            out[k] = v
        return out


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """
    Randomly flip image horizontally with option to use same choice on mask

    Args: p (float): probability of the image being flipped. Default value is 0.5

    Note: image should be flipped using `forward` and mask using `mask_fn`
    """

    def __init__(self, p=0.5):
        super().__init__(p)
        self._current_state = None

    def forward(self, x):
        return self.__call__(x)

    def __call__(self, x, state=None):
        if state is None:
            self._current_state = random.random() > (1.0 - self.p)
            state = self._current_state
        if state:
            x = F.hflip(x)
        return x

    def mask_fn(self, x):
        return self.__call__(x, state=self._current_state)


class Div255:
    def __call__(self, img):
        return img.clip(0, 255).div(255)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToTensor(transforms.ToTensor):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 3:
            return torch.from_numpy(x).permute(2, 0, 1)
        return super().__call__(x)


class Normalize(transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)
        self.ndim = len(mean)

    def __call__(self, x):
        if self.ndim == 1 or self.ndim == x.size(0):
            return super().__call__(x)
        if not self.inplace:
            x = x.clone()
        x[: self.ndim] = super().__call__(x[: self.ndim])
        return x
