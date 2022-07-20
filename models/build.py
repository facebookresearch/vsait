# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .discriminators import VSADiscriminator
from .generators import ResNetGenerator


def build_backbone(cfg):
    """Build backbone generators/discriminators for VSAIT model"""
    # ResNet generator
    g_net = ResNetGenerator(
        cfg.TASK_MODEL.IN_CHANNELS,
        cfg.TASK_MODEL.OUT_CHANNELS,
        cfg.TASK_MODEL.RES_CHANNELS,
        cfg.TASK_MODEL.RES_BLOCKS,
        **cfg.TASK_MODEL.get("GENERATOR_KWARGS", {}),
    )

    # VSA discriminator
    lsh_cfg = cfg.TASK_MODEL.get("LSH", {})
    d_net = VSADiscriminator(
        lsh_cfg.get("out_channels"),
        num_scales=cfg.TASK_MODEL.get("DISCRIMINATOR_SCALES", 1),
        **cfg.TASK_MODEL.get("DISCRIMINATOR_KWARGS", {}),
    )

    return g_net, d_net
