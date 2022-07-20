# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def parse_args_train():
    parser = argparse.ArgumentParser(
        description="Train VSAIT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=True,
        help="run name",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/vsait.yaml",
        help="yaml model config file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="./configs/gta2cityscapes.yaml",
        help="yaml data config file",
    )
    parser.add_argument(
        "--aug_config",
        type=str,
        default="./configs/gta2cityscapes_aug.yaml",
        help="json augmentation file",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=5000,
        help="Number of training batches between val loops",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=None,
        help="Number of training epochs between val loops (overrides val_check_interval)",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="proportion of validation set to use",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200000,
        help="Number of total training batches for the training run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/",
        help="path to local directory to save results",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="checkpoint passed to Trainer as `resume_from_checkpoint`",
    )

    args = parser.parse_args()
    return args


def parse_args_test():
    parser = argparse.ArgumentParser(
        description="Adapt using trained VSAIT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=True,
        help="run name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to checkpoint of trained model",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/vsait.yaml",
        help="yaml model config file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="./configs/gta2cityscapes.yaml",
        help="yaml data config file",
    )
    parser.add_argument(
        "--aug_config",
        type=str,
        default="./configs/gta2cityscapes_aug.yaml",
        help="json augmentation file",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
    parser.add_argument(
        "--limit_test_batches",
        type=float,
        default=1.0,
        help="proportion of test set to adapt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/",
        help="path to local directory to save adapted images",
    )

    args = parser.parse_args()
    return args
