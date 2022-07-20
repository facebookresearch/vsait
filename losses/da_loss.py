# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


class GANLoss(torch.nn.Module):
    """
    Generative Adversarial Network (GAN) losses

    Args:
        cfg: Config with the following nodes:
            ```
            TASK_MODEL.LOSSES : list
                losses to use during training. [default: ['adv','discr']]
                See notes for more detail.
            TASK_MODEL.[LOSS_NAME]_LOSS_COEF : float
                optional coefficients for losses used in GAN
                [default: 1. for all losses]
            TASK_MODEL.LABEL_SMOOTHING : float
                optional label-smoothing alpha value [default: 0.]
            TASK_MODEL.GAN_LOSS_FN : str
                GAN loss function to use. one of ['loss_ls_gan','loss_bce_gan','loss_hinge_gan']
                (least-squares or binary cross entropy, respectively)
                [default: 'loss_ls_gan']
            ```
    """

    def __init__(self, cfg):
        super(GANLoss, self).__init__()
        self.losses = self._get_losses(cfg)
        self.weight_dict = self._get_weight_dict(cfg)
        self.alpha = cfg.TASK_MODEL.get("LABEL_SMOOTHING", 0.0)
        self.gan_loss_fn = cfg.TASK_MODEL.get("GAN_LOSS_FN", "loss_hinge_gan")
        assert hasattr(self, self.gan_loss_fn), (
            'No loss function named "%s" found.' % self.gan_loss_fn
        )
        self.log = {}

        # get loss map
        self.loss_map = self._get_loss_map()

    def _get_losses(self, cfg):
        """Get losses used in GAN model"""
        losses = cfg.TASK_MODEL.get("LOSSES", ["adv", "discr"])
        return losses

    def _get_weight_dict(self, cfg):
        """Get weights for losses used in GAN model"""
        # init weights for each loss as 1.
        weight_dict = {}.fromkeys(self.losses, 1.0)
        for loss_name in self.losses:
            # check if config has loss coef for loss name
            cfg_loss_name = loss_name.upper() + "_LOSS_COEF"
            if cfg.TASK_MODEL.get(cfg_loss_name) is not None:
                weight_dict.update({loss_name: cfg.TASK_MODEL.get(cfg_loss_name)})

        return weight_dict

    def _weight_list(self, weight, n):
        """set weight as list with `len(weight) == n` if not already"""
        if not isinstance(weight, list):
            weight = [weight]
        if len(weight) < n:
            weight = (weight * n)[:n]
        return weight

    def label_smoothing(self, label, alpha=0.0):
        """label smoothing for gan losses"""
        return (1.0 - alpha) * label + alpha / 2.0

    def loss_bce_gan(self, output, label, weight=1.0):
        """Cross entropy GAN loss with optional label-smoothing (outputs should be pre-sigmoid)"""
        weight = self._weight_list(weight, len(output))

        loss = 0
        cnt = 0
        for o, la, w in zip(output, label, weight):
            la = self.label_smoothing(la, self.alpha)

            # bce loss between output and label
            loss_i = F.binary_cross_entropy_with_logits(
                o, torch.empty_like(o).fill_(la)
            )

            loss = loss + loss_i * w
            cnt += 1

        return loss / max(cnt, 1)

    def loss_ls_gan(self, output, label, weight=1.0):
        """LS-GAN loss with optional label-smoothing"""
        weight = self._weight_list(weight, len(output))

        loss = 0
        cnt = 0
        for o, la, w in zip(output, label, weight):
            la = self.label_smoothing(la, self.alpha)

            # mse loss between output and label
            loss_i = F.mse_loss(o, torch.empty_like(o).fill_(la))

            loss = loss + loss_i * w
            cnt += 1

        return loss / max(cnt, 1)

    def loss_hinge_gan(self, output, label, weight=1.0):
        """Hinge GAN loss"""
        weight = self._weight_list(weight, len(output))

        # if all labels==1, adversarial
        if all(label):
            adv = True
        else:
            adv = False

        loss = 0
        cnt = 0
        for o, la, w in zip(output, label, weight):
            # adversarial training
            if adv:
                loss_i = -torch.mean(o)
            elif la == 1:
                # real (discriminator)
                minval = torch.min(o - 1.0, torch.zeros_like(o))
                loss_i = -torch.mean(minval)
            else:
                # fake (discriminator)
                minval = torch.min(-o - 1.0, torch.zeros_like(o))
                loss_i = -torch.mean(minval)

            loss = loss + loss_i * w
            cnt += 1

        return loss / max(cnt, 1)

    def _get_loss_map(self):
        """Mapping for losses for basic GAN: [adv, discr]"""
        return {
            # gan losses
            "adv": getattr(self, self.gan_loss_fn),
            "discr": getattr(self, self.gan_loss_fn),
        }

    def forward(self, outputs, targets):
        """
        Compute losses for given model

        Args:
            outputs: dictionary of outputs for for chosen losses
                (e.g., `{'task': task_out, 'adv': d_out}`)
            targets: dictionary of targets for corresponding losses in `outputs`
                (e.g., `{'task': task_target, 'adv': 1.}`)
        """
        loss = 0
        self.log = {}
        for loss_name in self.losses:
            # if loss not found in outputs, skip
            if outputs.get(loss_name) is None:
                continue

            # get output/target associated with loss_name
            output_i = outputs.get(loss_name)
            target_i = targets.get(loss_name)

            # if weight is list, apply within function
            weight = self.weight_dict.get(loss_name, 1.0)
            if isinstance(weight, list):
                loss_i = self.loss_map[loss_name](output_i, target_i, weight)
                weight = 1.0
            else:
                loss_i = self.loss_map[loss_name](output_i, target_i)

            # update log
            self.log.update({loss_name: loss_i.item()})

            # update loss
            loss = loss + loss_i * weight

        # if 0, return torch.zeros(1, requires_grad=True)
        if isinstance(loss, (int, float)):
            return torch.zeros(1, requires_grad=True)
        return loss


class VSALoss(GANLoss):
    """
    Loss class for VSAIT model
    """

    def __init__(self, cfg):
        super(VSALoss, self).__init__(cfg)
        self.cfg = cfg

        self.loss_map.update({"vsa": self.loss_vsa})

    def loss_vsa(self, output, target):
        """Minimize cosine distance between output/target"""
        loss = torch.mean(
            1.0
            - torch.cosine_similarity(
                output["source"], target["source"].detach(), dim=1
            )
        )

        return loss

    def _get_losses(self, cfg):
        """Get losses used in VSA model"""
        losses = cfg.TASK_MODEL.get("LOSSES", ["vsa", "adv", "discr"])
        return losses
