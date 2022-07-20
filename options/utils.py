# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


def get_full_output_dir(output_dir):
    # make dirs
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
