# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .dataset import LocalDataset
from .joint import JointDataLoader
from .loader import LocalLoaderTest, LocalLoaderTrain


def build_dataset(data_cfg):
    return LocalDataset(data_cfg)


def build_dataloader(cfg, is_train):
    dataset = build_dataset(cfg)
    if is_train:
        loader = LocalLoaderTrain(cfg)
    else:
        loader = LocalLoaderTest(cfg)
    dataloader = loader.load(dataset)
    return dataloader


def build_joint_dataloader(source_cfg, target_cfg, data_cfg, is_train):
    """Build a JointDataLoader from source and target configs."""
    # create source/target data loaders
    source_dataloader = build_dataloader(source_cfg, is_train)
    target_dataloader = build_dataloader(target_cfg, is_train)

    # build the joint loader
    joint_dataloader = JointDataLoader(data_cfg)
    joint_dataloader.build(source_dataloader, target_dataloader, is_train)

    return joint_dataloader
