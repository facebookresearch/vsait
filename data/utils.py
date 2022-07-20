# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torchvision.transforms as transforms
import yaml

from . import augmentations as core_aug


def get_augmentation(func):
    """a simple function str -> function"""
    # give core_aug precedence over torchvision.transforms
    aug = getattr(core_aug, func, None)
    if aug is None:
        aug = getattr(transforms, func, None)
    if aug is None:
        raise NameError(f"Augmentation {func} does not exist")
    return aug


def parse_aug(filename):
    """augmentations are yaml files
    args: filename [str] filename of yaml file
    returns: augmentation functions
    """
    assert os.path.isfile(filename), f"{filename} not found."

    # load data from yaml
    with open(filename, "r") as f:
        augmentations_str = yaml.load(f, Loader=yaml.SafeLoader)

    # get from optional AUG key
    if augmentations_str.get("AUG") is not None:
        augmentations_str = augmentations_str.get("AUG")

    assert "TRAIN" in augmentations_str, "You need to define the training augmentations"
    aug_dic = {}
    for split, augs in augmentations_str.items():
        augmentations = []
        for func in augs:
            if isinstance(func, dict):
                # kwarg/arg format func(**params) or func(*parms)
                for k, v in func.items():
                    augmentation = get_augmentation(k)
                    if isinstance(v, dict):
                        augmentations.append(augmentation(**v))
                    elif isinstance(v, list):
                        augmentations.append(augmentation(*v))
            elif isinstance(func, str):
                # no-arg format func()
                augmentation = get_augmentation(func)
                augmentations.append(augmentation())
            else:
                raise Exception(
                    f"Expected str or dict definition of function and arguments. Found: {type(func)}"
                )

        aug_dic[split] = augmentations

    # if validation augmentations are not defined, use the train augmentations
    if "VAL_TARGET" not in aug_dic:
        aug_dic["VAL_TARGET"] = aug_dic.get("TRAIN_TARGET") or aug_dic["TRAIN"]
    if "VAL_SOURCE" not in aug_dic:
        aug_dic["VAL_SOURCE"] = aug_dic["TRAIN"]
    return aug_dic
