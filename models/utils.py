# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re

import torch


def parse_batch(batch, keys=None):
    """
    Parse batch for keys e.g. 'image', 'label'

    Args:
        batch: batch to parse (e.g., `[{'image': torch.Tensor, 'label': torch.Tensor}]`)
        keys: keys to parse from batch (if batch contains dict components), otherwise
            if batch is not a torch.Tensor outputs are returned as `batch[:len(keys)]`

    Note:
        If `isinstance(batch, torch.Tensor)`, `batch` is returned. Additionally, if
        `batch` is a dict with keys 'source' or 'target', a dict of outputs like
        `{'source': outputs, 'target': outputs}` is returned.
    """
    keys = keys or ["image", "target"]
    assert isinstance(keys, list)
    outputs = {}

    # return batch if not dict
    if not isinstance(batch, dict):
        return batch

    # if all values are dict, return dict format
    if all(isinstance(v, dict) for v in batch.values()):
        for k, v in batch.items():
            values = [v.get(key) for key in keys]
            outputs[k] = values[0] if len(values) == 1 else values
        return outputs

    # otherwise get each key from batch dict
    outputs = [batch.get(k) for k in keys]
    return outputs[0] if len(outputs) == 1 else outputs


def parse_list_var(var, n):
    # return var if list, otherwise return [var]*n
    if isinstance(var, list):
        assert len(var) == n
        return var
    return [var] * n


def convert_str_args(args):
    # convert comma-separated str vars to bool/int/float if possible
    if args is None:
        return []

    out = []
    # remove parentheses
    args = args.replace("(", "").replace(")", "")
    if not args:
        return []

    # for each comma-separated arg try convert to bool/int/float
    for arg in args.split(","):
        arg = arg.strip()
        if arg.lower() in ["false", "true"]:
            # bool
            out.append(bool(["false", "true"].index(arg.lower())))
        elif arg.isdigit():
            # int
            out.append(int(arg))
        elif not arg.isalpha():
            # float
            try:
                out.append(float(arg))
            except ValueError:
                out.append(arg)
    return out


def parse_str_var(var):
    # convert_str_args if str and contains 'nn', otherwise return var
    if isinstance(var, str) and (var.find("nn") != -1 or var == "None"):
        if var == "None":
            return None

        # parse module/args from var
        mod_str, args_str = re.match(
            r"(torch.)?nn.([\w\d]+)(\(?[^\)]*\)?)", var
        ).groups()[1:]
        mod = getattr(torch.nn, mod_str)

        if args_str:
            return mod(*convert_str_args(args_str))

        return mod

    return var


def parse_dict_var(var, n):
    # for each item in dict, return v[n] if list else v
    assert isinstance(var, dict), f"expected type dict but found: {type(var)}"
    out = {}
    for k, v in var.items():
        if isinstance(v, list):
            out.update({k: v[n]})
        else:
            out.update({k: v})
    return out


def make_list(x):
    # return list(x.values()) if dict else [x] if not list
    if isinstance(x, dict):
        return list(x.values())
    elif not isinstance(x, list):
        return [x]
    return x
