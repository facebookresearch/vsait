# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os
import warnings
from io import BytesIO

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from data.build import build_joint_dataloader
from losses.da_loss import VSALoss

from models import utils
from models.iit import VSAIT
from PIL import Image
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torchvision.utils import make_grid
from utils.logger import get_module_logger

logger = get_module_logger(__name__)


class VSAITSolver(pl.LightningModule):
    """
    Solver class for VSAIT
    """

    # set random colors for logging mask images
    RAND_COLORS = torch.randn(1, 3, 256)

    def __init__(self, cfg):
        super(VSAITSolver, self).__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR
        self.model = VSAIT(cfg)
        self.criterion = VSALoss(cfg)
        self._load_model(cfg)

        # get image logging frequency preferences
        self.img_log_freq = getattr(self.cfg.TASK_MODEL, "IMAGE_LOGGING_FREQ", 0)
        self.val_img_log_freq = getattr(
            self.cfg.TASK_MODEL, "VAL_IMAGE_LOGGING_FREQ", self.img_log_freq
        )
        self.is_first_val_loop = True

    def _load_model(self, cfg):
        # load model weights
        if cfg.TASK_MODEL.WEIGHTS:
            logger.info(f"Loading weights from {cfg.TASK_MODEL.WEIGHTS}")
            loaded_model = pl_load(
                cfg.TASK_MODEL.WEIGHTS,
                map_location="cuda:0" if torch.cuda.is_available() else None,
            )
            loaded_model = loaded_model.get("state_dict", loaded_model)
            self.load_state_dict(loaded_model, strict=False)

    def _denormalize_image(
        self,
        img,
        mean=(0.5,),
        std=(0.5,),
        max_pixel_value=255.0,
    ):
        """denormalize image with shape (b, c, h, w) using mean/std"""
        b = img.size(0)
        if img.size(1) > 3:
            idx = (
                utils.mask_to_seg(img, ignore_index=0, dim=1)
                .view(b, 1, -1)
                .expand(-1, 3, -1)
                .cpu()
            )
            img = torch.gather(self.RAND_COLORS.expand(b, -1, -1), 2, idx).view(
                b, 3, *img.shape[-2:]
            )

        img = img.detach().cpu()
        img = img * torch.tensor(std).view(1, -1, 1, 1)
        img = img + torch.tensor(mean).view(1, -1, 1, 1)
        img = img * max_pixel_value
        return img

    def _log_images(self, img_dict, tags=None, img_size=None):
        """
        Log images to tensorboard with given tags

        Args:
            img_dict: dictionary of images, where the keys should startwith
                corresponding tag
            tags: tags for logging images whose keys startwith each tag
                if None, tags = list(img_dict.keys())
            img_size: image size to resize before logging
                if None, images are resized to first image size per tag
        """
        # if logger.experiment doesn't have "add_images", return
        if not self.logger or not hasattr(self.logger.experiment, "add_images"):
            return

        # log images for each tag
        for tag in tags or img_dict.keys():
            # resize/denormalize image using mean=0.5, std=0.5
            images = []
            for k, v in img_dict.items():
                if not k.startswith(tag):
                    continue

                if img_size or len(images) > 0:
                    v = F.interpolate(
                        v, img_size or images[0].shape[-2:], mode="nearest"
                    )

                image_i = self._denormalize_image(
                    v.detach(),
                    mean=(0.5,),
                    std=(0.5,),
                    max_pixel_value=1.0,
                )
                images.append(image_i)

            # make grid of images with max(2, batch_size) per row
            img = make_grid(
                torch.cat(images), nrow=max(2, images[0].size(0))
            ).unsqueeze(0)

            self.logger.experiment.add_images(tag, img, self.global_step)

    def _add_idx_to_filename(self, filename, image_idx):
        if filename is None:
            return image_idx

        out = []
        for i, fname in zip(image_idx, filename):
            out.append(fname.replace(".", f"_{i}."))
        return out

    def _save_image(self, img, output_path, height=None, width=None, img_ext=".png"):
        # create PIL image from numpy array
        img = Image.fromarray(img.permute(1, 2, 0).numpy().astype("uint8"))

        # resize image
        if width and height:
            img.resize((width, height), Image.NEAREST)

        # save file to manifold, otherwise locally
        with open(output_path + img_ext, "wb") as f:
            output = BytesIO()
            img.save(output, format="PNG")
            f.write(output.getvalue())

    def _save_images(
        self, img, filename, height=None, width=None, batch_idx=None, prefix=None
    ):
        if not isinstance(filename, list):
            filename = [filename]

        if len(img) != len(filename):
            print(
                f"number of images {len(img)} does not match number of filenames {len(filename)}."
            )

        # save image(s)
        for fname, img_i in zip(filename, img):
            # get filename
            if fname is not None:
                basename, ext = os.path.splitext(os.path.basename(fname))
            else:
                basename = batch_idx
                ext = ".png"
            # add prefix
            if prefix:
                basename = f"{prefix}_{basename}"
            output_path = os.path.join(self.output_dir, "images", basename)

            # denormalize/save image
            img_i.unsqueeze_(0)
            img_i = self._denormalize_image(
                img_i.detach(), mean=(0.5,), std=(0.5,)
            ).squeeze()

            self._save_image(img_i, output_path, height, width, ext)

    def forward(self, x, **kwargs):
        if self.training:
            return self.model(x, **kwargs)

        # if not training, inference call
        return self.model.inference(x)

    def setup(self, *args, **kwargs):
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        outputs, targets = self.forward(batch, optimizer_idx=optimizer_idx)

        loss = self.criterion(outputs, targets)

        # log scalars
        to_log = getattr(self.criterion, "log", {})
        to_log.update({"loss": loss})

        if self.logger:
            self.logger.log_metrics(to_log, self.global_step)

        # log images
        if (
            optimizer_idx in [0, None]
            and self.img_log_freq
            and batch_idx % self.img_log_freq == 0
        ):
            batch_imgs = utils.parse_batch(batch, ["image"])
            if not isinstance(batch_imgs, dict):
                batch_imgs = {"source": batch_imgs}

            tags = list(batch_imgs.keys())

            if hasattr(self.model, "fakes"):
                translated_imgs = {
                    "%s_translated" % k: v for k, v in self.model.fakes.items()
                }
                tags.extend(self.model.fakes.keys())
                batch_imgs.update(translated_imgs)

            self._log_images(batch_imgs, list(set(tags)))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # parse batch for image tensors
        src_img, filename = utils.parse_batch(batch["source"], ["image", "filename"])
        height, width = src_img.shape[-2:]
        tgt_img, tgt_filename = utils.parse_batch(
            batch["target"], ["image", "filename"]
        )
        tgt_height, tgt_width = tgt_img.shape[-2:]

        # append image index to filenames
        image_idx = [
            str(batch_idx * src_img.size(0) + n) for n in range(src_img.size(0))
        ]
        filename = self._add_idx_to_filename(filename, image_idx)
        tgt_filename = self._add_idx_to_filename(tgt_filename, image_idx)

        # generate translated source image
        img = self.model.inference(batch)

        # save source/translated images
        if self.val_img_log_freq and batch_idx % self.val_img_log_freq == 0:
            if self.is_first_val_loop:
                # save source/target on first loop only
                self._save_images(
                    src_img, filename, height, width, batch_idx, prefix="source"
                )
                self._save_images(
                    tgt_img,
                    tgt_filename,
                    tgt_height,
                    tgt_width,
                    batch_idx,
                    prefix="target",
                )

            # save translated images
            self._save_images(
                img,
                filename,
                height,
                width,
                batch_idx,
                prefix=f"translated_epoch={self.current_epoch}-step={self.global_step}",
            )
        return

    def validation_epoch_end(self, *args, **kwargs):
        self.is_first_val_loop = False

    def test_step(self, batch, batch_idx):
        # parse batch for image tensors
        src_img, filename = utils.parse_batch(batch["source"], ["image", "filename"])
        height, width = src_img.shape[-2:]

        # generate image from batch
        img = self.model.inference(batch)

        # save images
        self._save_images(img, filename, height, width, batch_idx)

    def train_dataloader(self):
        return build_joint_dataloader(
            self.cfg.DATA.TRAIN, self.cfg.DATA.TRAIN_TARGET, self.cfg.DATA, True
        )

    def val_dataloader(self):
        return build_joint_dataloader(
            self.cfg.DATA.VAL_SOURCE,
            self.cfg.DATA.VAL_TARGET,
            self.cfg.DATA,
            False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def _get_parameters(self, cfg):
        """helper function to prepare parameter dicts for optimizer"""
        # set parameter-specific groups
        param_group_names = [
            k.split("LR_")[-1] for k in cfg.keys() if k.startswith("LR_")
        ]

        param_dicts = [{"params": []}]
        param_dicts.extend(
            [
                {"params": [], "name": name, "lr": cfg.get(f"LR_{name}")}
                for name in param_group_names
            ]
        )

        # get named_parameters
        named_parameters = [
            (n, p)
            for n, p in self.named_parameters()
            if any(
                n.startswith(net_prefix)
                for net_prefix in cfg.get("NET_PREFIX", "").split(",")
            )
        ]
        if len(named_parameters) == 0:
            warnings.warn(
                "No parameters found for network prefix: %s" % cfg.get("NET_PREFIX", "")
            )

        for n, p in named_parameters:
            # skip params without requires_grad True
            if not p.requires_grad:
                continue

            # add parameters to base params
            if all(k not in n for k in param_group_names):
                param_dicts[0].get("params").append(p)
            else:
                # add to specific parameter group
                idx = [i + 1 for i, k in enumerate(param_group_names) if k in n][0]
                param_dicts[idx].get("params").append(p)

        return param_dicts

    def _build_optimizer(self, cfg):
        """helper function to build optimizer"""
        # get optimizer fn and kwargs from cfg
        opt_fn = getattr(torch.optim, cfg.get("NAME"))
        opt_argnames = inspect.getfullargspec(opt_fn).args
        opt_kwargs = {k.lower(): v for k, v in cfg.items() if k.lower() in opt_argnames}

        # get parameters
        param_dicts = self._get_parameters(cfg)

        # set optimizer
        return opt_fn(param_dicts, lr=cfg.get("LEARNING_RATE"), **opt_kwargs)

    def _build_scheduler(self, sched_cfg, opt_cfg, optimizer):
        """helper function to build scheduler"""
        # if no scheduler defined, skip
        if sched_cfg.get("NAME") is None:
            return None

        # get scheduler fn and kwargs from cfg
        sched_fn = getattr(torch.optim.lr_scheduler, sched_cfg.get("NAME"))
        sched_argnames = inspect.getfullargspec(sched_fn).args
        sched_kwargs = {
            k.lower(): v for k, v in sched_cfg.items() if k.lower() in sched_argnames
        }

        # set generator and discriminator lr schedulers
        if opt_cfg.get("NET_PREFIX", ""):
            sched_name = "_".join(
                ["lr", opt_cfg.get("NAME"), opt_cfg.get("NET_PREFIX", "")]
            )
        else:
            sched_name = "_".join(["lr", opt_cfg.get("NAME")])

        return {
            "scheduler": sched_fn(optimizer, **sched_kwargs),
            "interval": sched_cfg.get("INTERVAL", "epoch"),
            "name": sched_name,
        }

    def configure_optimizers(self):
        """
        Configure optimizers/schedulers for training

        config should contain an OPTIMIZER(S) node with the following structure:
            NAME: torch.optim name (e.g., Adam)
            NET_PREFIX: prefix network name for selecting parameters (e.g., `model.g_net`)
                [default: '' (all parameters)]
            LEARNING_RATE: learning rate for optimizer
            LR_[parameter]: optional parameter-specific learning rates, where parameters
                are included if their name contains [parameter] (e.g., `LR_backbone: 0.0001`)
            [kwarg]: other keyword arguments passed to optimizer at init
                (e.g., `BETAS: [0.5, 0.999]`)

        for the schedulers, a SCHEDULER(S) node should have a similar structure:
            NAME: torch.optim.lr_scheduler name (e.g., ExponentialLR)
            INTERVAL: optional interval for stepping scheduler (PL default is 'epoch')
            [kwargs]: other keyword arguments passed to optimizer at init
                (e.g., `GAMMA: 0.95`)

        Note:
            For multiple optimizers/schedulers, use OPTIMIZERS or SCHEDULERS instead with
            a list with above structure for each optimizer/scheduler. For VSAIT, two
            optimizers are expected with one corresponding to the discriminator network
            and the other corresponding to the generator (set using `NET_PREFIX`).
        """
        # get optimizer config (fall back to OPTIMIZER)
        opt_cfg = self.cfg.get("OPTIMIZERS") or self.cfg.get("OPTIMIZER", {})
        if not isinstance(opt_cfg, list):
            opt_cfg = [opt_cfg]

        # get scheduler config (fall back to SCHEDULER)
        sched_cfg = self.cfg.get("SCHEDULERS") or self.cfg.get("SCHEDULER", {})
        if not isinstance(sched_cfg, list):
            sched_cfg = [sched_cfg]
        sched_cfg = (sched_cfg * len(opt_cfg))[: len(opt_cfg)]

        # build optimizers, schedulers
        optimizers = []
        schedulers = []
        for opt_cfg_i, sched_cfg_i in zip(opt_cfg, sched_cfg):
            # build optimizer
            optimzier_i = self._build_optimizer(opt_cfg_i)
            optimizers.append(optimzier_i)

            # build scheduler
            scheduler_i = self._build_scheduler(sched_cfg_i, opt_cfg_i, optimzier_i)
            if scheduler_i is not None:
                schedulers.append(scheduler_i)

        if len(schedulers) == 0:
            return optimizers
        return optimizers, schedulers
