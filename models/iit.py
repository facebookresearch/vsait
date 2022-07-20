# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import utils
from .build import build_backbone
from .feature_extractor import build_feature_extractor
from .lsh import LSHCat
from .networks import MLP


class IIT(torch.nn.Module):
    """
    Image-to-image translation

    Generic iit model
    """

    def __init__(self, cfg):
        super(IIT, self).__init__()
        self.cfg = cfg
        self.g_net, self.d_net = build_backbone(cfg)
        self.fakes = {}

    def parse_optimizer_key(self, optimizer_idx, key_mapping=None):
        key_mapping = key_mapping or {"g_net": "adv", "d_net": "discr"}
        # if single optimizer, return adv
        if optimizer_idx is None:
            return "adv"
        # get net_prefixes
        if not hasattr(self, "net_prefix"):
            self.net_prefix = [
                v.get("NET_PREFIX", "") for v in self.cfg.get("OPTIMIZERS", [{}])
            ]
        # search for keys and return value if found in net prefix
        for k, v in key_mapping.items():
            if self.net_prefix[optimizer_idx or 0].find(k) > -1:
                return v
        raise Exception(
            "Could not determine optimizer from prefixes: %a" % self.net_prefix
        )

    def inference(self, batch):
        """Generate translated image from batch"""
        if isinstance(batch, dict) and "source" in batch:
            x = utils.parse_batch(batch["source"], ["image"])
        else:
            x = utils.parse_batch(batch, ["image"])
        return self.generate(x)

    def generate(self, x):
        """
        Generate translated image
        """
        return self.g_net(x)

    def generate_batch(self, batch, key):
        pass

    def discriminate(self, x, detach=False):
        pass

    def discriminate_batch(self, batch, detach=False):
        pass

    def forward(self, batch, optimizer_idx=0):
        pass


class VSAIT(IIT):
    """
    Vector Symbolic Architecture Image Translation (VSAIT)

    This approach uses the VSA framework to learn a translation network
    that approximates the following VSA manipulation:

        `X_{translated} = X_{source} * S_{source} * S_{target}`

    where `X` is the high-dimensional vector representation of image features
    and `S` is the high-dimensional vector representation of source or target
    domain-specific information.

    Note:
        This approach requires a `FEATURE_EXTRACTOR` and `LSH` nodes in `TASK_MODEL`
        of the config. These nodes define the pre-trained feature extractor and
        locality-sensitive hashing networks used to project image features into the VSA
        hyperspace. See `feature_extractor.py` and `networks.py` for more information.
    """

    def __init__(self, cfg):
        super(VSAIT, self).__init__(cfg)

        # set in channels for feature extractor dummy input
        fx_key = "FEATURE_EXTRACTOR"
        self.in_channels = cfg.TASK_MODEL.IN_CHANNELS
        dummy_in_channels = cfg.TASK_MODEL.get(fx_key).get(
            "IN_CHANNELS", self.in_channels
        )

        # get vgg network (or torch network)
        self.feature_extractor = build_feature_extractor(cfg.TASK_MODEL.get(fx_key))

        # get feature map shapes from feature extractor
        dummy_output = utils.make_list(
            self.feature_extractor(torch.zeros(1, dummy_in_channels, 256, 256))
        )
        feat_shapes = [tuple(o.shape) for o in dummy_output]

        print(f"Feature extractor output shapes: {feat_shapes}")

        # set in channels for LSH
        self.feat_in_channels = [f_s[1] for f_s in feat_shapes]

        # create lsh for each feat output
        self.lsh_feat = LSHCat(self.feat_in_channels, **cfg.TASK_MODEL.get("LSH"))

        # build network for generating src-to-tgt style vector
        map_net_kwargs = cfg.TASK_MODEL.get("MAP_NET_KWARGS", {})
        map_net_kwargs.setdefault(
            "n_channels", [self.lsh_feat.out_channels, self.lsh_feat.out_channels]
        )
        map_net_kwargs.setdefault("kernel_size", 1)
        map_net_kwargs.setdefault("padding", 0)
        self.g_net_map = torch.nn.Sequential(
            MLP(
                self.lsh_feat.out_channels,
                conv=True,
                **map_net_kwargs,
            ),
            torch.nn.Tanh(),
        )
        # init last layer to 0, 1 for identity
        torch.nn.init.constant_(self.g_net_map[0][-1].conv.weight, 0.0)
        torch.nn.init.constant_(self.g_net_map[0][-1].conv.bias, 1.0)

    def generate_batch(self, batch, key):
        """Generate outputs, targets for training"""
        outputs = {"vsa": {}}
        targets = {"vsa": {}}
        self.fakes = {}

        # generate masked/translated image for source/target
        for k, v in batch.items():
            # parse batch for image, semantic label
            x = utils.parse_batch(v, ["image"])

            # skip target domain
            source = k == "source"
            if not source:
                continue

            # generate fake
            fake = self.generate(x)
            self.fakes.update({k: fake})

            # skip discr (keep targets dict)
            if key == "discr":
                outputs = {}
                return outputs, targets

            # extract features from image
            fake = utils.make_list(self.feature_extractor(fake, "target"))
            src = utils.make_list(self.feature_extractor(x, "source"))

            # get fake, src vectors
            fake = self.lsh_feat(fake)
            src = self.lsh_feat(src)

            # generate mapping network
            src_to_tgt = self.g_net_map(src)

            # vsa loss
            outputs.get("vsa").update({k: fake * src_to_tgt.detach()})
            targets.get("vsa").update({k: src})

        return outputs, targets

    def discriminate(self, x, detach=False, **kwargs):
        """Discriminate input with option to detach and pass kwargs"""
        if detach:
            x = x.detach()
        return self.d_net(x, **kwargs)

    def discriminate_batch(self, batch, detach=False, **kwargs):
        """Discriminate batch of inputs with option to detach"""
        out = []
        for v in batch.values():
            # parse batch for image
            x = utils.parse_batch(v, ["image"])

            # discriminate
            d_out = self.discriminate(x, detach=detach, **kwargs)

            # extend or append to output
            if isinstance(d_out, list):
                out.extend(d_out)
            else:
                out.append(d_out)
        return out

    def forward(self, batch, optimizer_idx=0):
        # init output/target dicts
        outputs = {}
        targets = {}

        # generate translated image, get outputs/targets
        key = self.parse_optimizer_key(optimizer_idx)
        outputs, targets = self.generate_batch(batch, key)

        # discriminate
        d_out = self.discriminate_batch(
            {"source": self.fakes["source"]},
            detach=(key == "discr"),
            src=utils.parse_batch(batch["source"], ["image"]),
            feature_extractor=self.feature_extractor,
            lsh_feat=self.lsh_feat,
            map_net=self.g_net_map,
        )
        labels = [1.0 if key == "adv" else 0.0] * len(d_out)

        if key == "discr":
            d_out.extend(
                self.discriminate_batch(
                    {"target": utils.parse_batch(batch["target"], ["image"])},
                    detach=True,
                    feature_extractor=self.feature_extractor,
                    lsh_feat=self.lsh_feat,
                )
            )
            labels.extend([1.0] * len(labels))

        outputs.update({key: d_out})
        targets.update({key: labels})

        return outputs, targets
