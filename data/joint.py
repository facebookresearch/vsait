# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


class JointDataLoader(object):
    """
    Class for loading data from two dataloaders jointly
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataloader_A = None
        self.dataloader_B = None
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = None
        self.is_train = None

    def build(
        self, dataloader_A, dataloader_B, is_train, max_dataset_size=float("inf")
    ):
        self.dataloader_A = dataloader_A
        self.dataloader_B = dataloader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size
        self.is_train = is_train

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.dataloader_A_iter = iter(self.dataloader_A)
        self.dataloader_B_iter = iter(self.dataloader_B)
        self.iter = 0
        # Set global seed to make deterministic for val
        if self.is_train is False:
            np.random.seed(0)
        return self

    def __len__(self):
        # for validation, use VAL_SOURCE as length
        if not self.is_train:
            return len(self.dataloader_A)
        return max(len(self.dataloader_A), len(self.dataloader_B))

    def __next__(self):
        A = None
        B = None
        try:
            A = next(self.dataloader_A_iter)
        except StopIteration:
            if A is None:
                self.stop_A = True
                self.dataloader_A_iter = iter(self.dataloader_A)
                A = next(self.dataloader_A_iter)

        try:
            B = next(self.dataloader_B_iter)
        except StopIteration:
            if B is None:
                self.stop_B = True
                self.dataloader_B_iter = iter(self.dataloader_B)
                B = next(self.dataloader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {
                "source": A,
                "target": B,
            }
