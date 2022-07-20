# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.models
from torchvision.models.feature_extraction import create_feature_extractor


class IntermediateOutputs(torch.nn.Module):
    """
    Return intermediate outputs from a network

    Args:
        net: network with intermediate layer to return
        return_nodes: dictionary of modules and output names
            (e.g., {"layer1.conv": "conv_features"}), list of module names (e.g., ["layer1.conv"]),
            or module name (e.g., "layer1.conv"). Output type will be torch.Tensor if `return_nodes`
            type is str, otherwise output type will be dict.
        requires_grad: Sets `net` parameters `requires_grad` attribute [default: False]

    Note:
        This approach registers a forward hook to store outputs, which may not work for all methods.
    """

    def __init__(self, net, return_nodes, requires_grad=False):
        super().__init__()
        if isinstance(return_nodes, list):
            return_nodes = {k: str(i) for i, k in enumerate(return_nodes)}
        elif not isinstance(return_nodes, dict):
            return_nodes = {return_nodes: "_"}

        for p in net.parameters():
            p.requires_grad_(requires_grad)

        self.net = net
        self.return_nodes = return_nodes

    def _get_module(self, net, name):
        mod = net
        for name_i in name.split("."):
            try:
                mod = getattr(mod, name_i)
            except Exception as msg:
                print(msg)
                raise Exception(net)
        return mod

    def init_hooks(self):
        self.hooks = {}
        self.outputs = {}
        self.mod_map = {}
        # register hook for each module
        for mod_name, out_name in self.return_nodes.items():
            mod = self._get_module(self.net, mod_name)
            self.mod_map.update({mod: out_name})

            def hook_fn(mod, input, output):
                self.outputs.update({self.mod_map[mod]: output})

            hook = mod.register_forward_hook(hook_fn)
            self.hooks.update({mod: hook})

    def remove_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}

    def forward(self, x):
        # initialize layer hooks
        self.init_hooks()

        # pass x through model
        try:
            self.net(x)
        finally:  # remove hooks
            self.remove_hooks()

        # return hook outputs
        out = self.outputs
        self.outputs = {}
        return out


class FeatureExtractor(torch.nn.Module):
    """
    Feature Extractor class for pre-trained torchvision models

    Args:
        cfg: Config containing following node:
        ```
        RESCALE: [mu, sigma]
        MODEL:
        ```
    """

    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.get("IN_CHANNELS")

        rescale = cfg.get("RESCALE", [])
        if rescale:
            self._set_rescale(rescale)

        self.net = self.build_model(cfg)

    def _set_rescale(self, rescale):
        # rescale should be {"source": [mu, sigma], "target": [mu, sigma]} or [mu, sigma]
        if isinstance(rescale, dict):
            n = 2
            src_mu, src_sigma = rescale.get("source", [[], []])
            tgt_mu, tgt_sigma = rescale.get("target", [[], []])
            mu = [src_mu, tgt_mu]
            sigma = [src_sigma, tgt_sigma]
            self._rescale_index = {"source": 0, "target": 1}
        else:
            n = 1
            mu, sigma = rescale
            self._rescale_index = {"source": 0, "target": 0}

        self.mu = torch.tensor(mu).view(n, 1, -1, 1, 1)
        self.sigma = torch.tensor(sigma).view(n, 1, -1, 1, 1)

    def build_pretrained_torchvision_model(
        self, model_name="inception_v3", weights="DEFAULT", **kwargs
    ):
        # identity model (Sequential)
        if model_name.lower() == "identity":
            return torch.nn.Sequential()

        # get pretrained model
        model_fn = getattr(torchvision.models, model_name, None)
        if model_fn is None:  # try detection
            model_fn = getattr(torchvision.models.detection, model_name, None)
        if model_fn is None:  # try segmentation
            model_fn = getattr(torchvision.models.segmentation, model_name, None)
        assert model_fn is not None, (
            "Unable to find MODEL %s in `torchvision.models`." % model_name
        )
        print("[%s] Loading MODEL: %s" % (self.__class__.__name__, model_name))

        # try building model with `weights` arg, fallback to `pretrained`
        try:
            return model_fn(weights=weights, **kwargs)
        except Exception as msg:
            pretrained = weights is not None
            print(
                f"[{self.__class__.__name__}] Error loading weights: {weights}. "
                f"Trying with `pretrained={pretrained}`.\n{msg}"
            )
        return model_fn(pretrained=pretrained, **kwargs)

    def build_model(self, cfg):
        """
        Build model using config file to obtain:
            IN_CHANNELS: (optional) number of input channels for network,
            RESCALE: (optional) list of `[mu, sigma]` parameters for normalization,
            MODEL: model name (from `torchvision.models`),
            RETURN_NODES: layer(s) at which distance metric is computed,
            **kwargs: keyword arguments passed when getting the model

        The MODEL field and any keyword arguments are passed to the appropriate
        `torchvision.models` (including detection and segmentation modules)
        function to get the model. The resulting model and RETURN_NODES field are
        passed to `IntermediateOutputs`.

        If MODEL is "identity", `torch.nn.Sequential()` is returned.
        """
        self.model_name = cfg.get("MODEL")
        return_nodes = cfg.get("RETURN_NODES")
        kwargs = {
            k.lower(): v
            for k, v in cfg.items()
            if k
            not in [
                "IN_CHANNELS",
                "MODEL",
                "RESCALE",
                "RETURN_NODES",
            ]
        }
        kwargs.setdefault("weights", "DEFAULT")

        # build inception_v3 if no model provided (default)
        if self.model_name is None:
            return_nodes = {"avgpool": "output"}
            self.model_name = "inception_v3"
        elif self.model_name.lower() == "identity":
            return_nodes = None
            kwargs = {}

        # build network
        net = self.build_pretrained_torchvision_model(self.model_name, **kwargs)

        # set network to evaluate mode
        net.eval()

        # set return_nodes
        if return_nodes is None:
            self.return_nodes = {"": ""}
            return net
        elif isinstance(return_nodes, str):
            return_nodes = {return_nodes: ""}
        elif isinstance(return_nodes, list):
            return_nodes = {k: str(n) for n, k in enumerate(return_nodes)}
        assert isinstance(
            return_nodes, dict
        ), "return_nodes should be a dict mapping layer names to output names"
        self.return_nodes = return_nodes

        # get intermediate output nodes
        print(
            "[%s] Using output layer(s): %s"
            % (self.__class__.__name__, list(return_nodes.keys()))
        )
        try:
            return create_feature_extractor(net, return_nodes)
        except Exception as msg:
            print(
                f"Encountered error with `create_feature_extractor`, using `IntermediateOutputs` instead: {msg}"
            )
        return IntermediateOutputs(net, return_nodes, requires_grad=False)

    def _rescale(self, x, domain="source"):
        if not hasattr(self, "mu") or not hasattr(self, "sigma"):
            return x

        # denormalize from (-1, 1)
        x = x * 0.5 + 0.5

        # if domain-specific stats, get index
        idx = self._rescale_index.get(domain, 0)

        # normalize using (mean, std)
        return (x - self.mu[idx].to(x.device)) / self.sigma[idx].to(x.device)

    def forward(self, x, domain="source"):
        x = self._rescale(x, domain)

        if self.in_channels is not None and self.in_channels > x.size(1):
            x = x.expand(-1, self.in_channels, -1, -1)

        return self.net(x)


def build_feature_extractor(cfg):
    """
    Build a feature extractor

    Torchvision models are defined using a `FEATURE_EXTRACTOR` node
    """
    return FeatureExtractor(cfg)
