# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytorch_lightning as pl

from data.utils import parse_aug
from options.parser import parse_args_test
from options.utils import get_full_output_dir

from solver import VSAITSolver
from utils.config import get_cfg
from utils.logger import get_module_logger


def main(args):
    logger = get_module_logger(__name__)
    logger.info("Parsing arguments")

    output_dir = os.path.join(args.output_dir, args.name)

    # get full_output_dir from logger and makedirs if needed
    full_output_dir = get_full_output_dir(output_dir)

    # parse config files
    cfg = get_cfg()
    cfg.merge_from_file(args.model_config)
    cfg.merge_from_file(args.data_config)
    logger.info(f"Loading model config: {args.model_config}")
    logger.info(f"Loading data config: {args.data_config}")
    cfg["args"] = vars(args)
    cfg.OUTPUT_DIR = full_output_dir
    cfg.DATA.VAL_SOURCE.BATCH_SIZE = args.batch_size
    cfg.DATA.VAL_TARGET.BATCH_SIZE = args.batch_size

    # load checkpoint
    if args.checkpoint:
        cfg.TASK_MODEL.WEIGHTS = args.checkpoint

    # parse augmentations
    augmentations = parse_aug(args.aug_config)
    logger.info(f"Using {args.aug_config} augmentations")
    cfg.DATA.TRAIN.AUGMENTATIONS = augmentations["TRAIN"]
    cfg.DATA.VAL_TARGET.AUGMENTATIONS = augmentations["VAL_TARGET"]
    cfg.DATA.VAL_SOURCE.AUGMENTATIONS = augmentations["VAL_SOURCE"]

    # build solver
    solver = VSAITSolver(cfg)

    # adapt
    trainer = pl.Trainer(
        limit_test_batches=args.limit_test_batches,
        gpus=args.num_gpus,
    )
    trainer.test(solver)


if __name__ == "__main__":
    # get arguments
    args = parse_args_test()

    main(args)
