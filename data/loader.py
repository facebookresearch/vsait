# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data


class LocalLoaderTrain:
    """
    Create local train dataloader with random sampling.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _create_sampler(self, dataset):
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, self.cfg.BATCH_SIZE, drop_last=True
        )
        return batch_sampler

    def load(self, dataset):
        batch_sampler = self._create_sampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.cfg.NUM_WORKERS,
            batch_sampler=batch_sampler,
        )


class LocalLoaderTest:
    """
    Create test data loader with sequential or random sampling
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _create_sampler(self, dataset):
        if self.cfg.get("SHUFFLE", False):
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, self.cfg.BATCH_SIZE, drop_last=False
        )
        return batch_sampler

    def load(self, dataset):
        batch_sampler = self._create_sampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.cfg.NUM_WORKERS,
            batch_sampler=batch_sampler,
        )
