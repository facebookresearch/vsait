# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl

from data.utils import parse_aug
from options.parser import parse_args_train
from options.utils import get_full_output_dir
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from solver import VSAITSolver
from utils.config import get_cfg
from utils.logger import get_module_logger


def main(args):
    logger = get_module_logger(__name__)
    logger.info("Parsing arguments")

    output_dir = args.output_dir

    pl_logger = pl.loggers.TensorBoardLogger(output_dir, args.name)

    full_output_dir = get_full_output_dir(pl_logger.log_dir)

    # pares config files
    cfg = get_cfg()
    cfg.merge_from_file(args.model_config)
    cfg.merge_from_file(args.data_config)
    logger.info(f"Loading model config: {args.model_config}")
    logger.info(f"Loading data config: {args.data_config}")
    cfg["args"] = vars(args)
    cfg.OUTPUT_DIR = full_output_dir
    cfg.DATA.TRAIN.BATCH_SIZE = args.batch_size
    cfg.DATA.TRAIN_TARGET.BATCH_SIZE = args.batch_size

    # parse augmentations
    augmentations = parse_aug(args.aug_config)
    logger.info(f"Using {args.aug_config} augmentations")
    cfg.DATA.TRAIN.AUGMENTATIONS = augmentations["TRAIN"]
    cfg.DATA.TRAIN_TARGET.AUGMENTATIONS = augmentations.get("TRAIN_TARGET")
    cfg.DATA.VAL_TARGET.AUGMENTATIONS = augmentations["VAL_TARGET"]
    cfg.DATA.VAL_SOURCE.AUGMENTATIONS = augmentations["VAL_SOURCE"]

    # build solver
    solver = VSAITSolver(cfg)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
    )
    callbacks = [checkpoint_callback]

    # check val every epoch or interval
    if args.check_val_every_n_epoch:
        check_val_epoch = True
    else:
        check_val_epoch = False

    # train
    trainer = pl.Trainer(
        logger=pl_logger,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        callbacks=callbacks,
        default_root_dir=output_dir,
        weights_save_path=output_dir,
        check_val_every_n_epoch=args.check_val_every_n_epoch if check_val_epoch else 1,
        val_check_interval=args.val_check_interval if not check_val_epoch else None,
        limit_val_batches=args.limit_val_batches,
        max_steps=args.max_steps,
        gpus=args.num_gpus,
    )
    trainer.fit(solver, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    # get arguments
    args = parse_args_train()

    main(args)
