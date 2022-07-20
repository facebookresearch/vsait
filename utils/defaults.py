# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .config import CfgNode as CN

_C = CN()

_C.VERSION = 0
_C.OUTPUT_DIR = ""

_C.TASK_MODEL = CN()
_C.TASK_MODEL.DISCRIMINATOR_SCALES = 1
_C.TASK_MODEL.IN_CHANNELS = 3
_C.TASK_MODEL.OUT_CHANNELS = 3
_C.TASK_MODEL.RES_CHANNELS = 64
_C.TASK_MODEL.RES_BLOCKS = 9
_C.TASK_MODEL.WEIGHTS = ""
