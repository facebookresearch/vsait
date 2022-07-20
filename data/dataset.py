# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from PIL import Image

from .augmentations import Compose


class BaseDataset(metaclass=ABCMeta):
    """Base dataset class"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.augmentations = Compose(cfg.AUGMENTATIONS)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class LocalDataset(BaseDataset):
    """
    Dataset for loading images/masks locally.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        data_dir = cfg.DATA_DIR
        ext = getattr(cfg, "EXT", ".png,.jpg,.jpeg")

        # get images paths
        self.images = []
        for ext_type in ext.split(","):
            self.images.extend(
                glob.glob(os.path.join(data_dir, "**", f"*{ext_type}"), recursive=True)
            )

    def __getitem__(self, index):
        # load image
        image_path = self.images[index]
        image = Image.open(image_path)
        image = image.convert("RGB")
        input_dict = {"image": image}

        # perform augmentations
        augmented = self.augmentations(**input_dict)

        # create output dict
        output_dict = {"filename": image_path}
        for k, v in augmented.items():
            v = torch.tensor(np.asarray(v))
            output_dict.update({k: v.unsqueeze(0) if v.ndim == 2 else v})

        return output_dict

    def __len__(self):
        return len(self.images)
