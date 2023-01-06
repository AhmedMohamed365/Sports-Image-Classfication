
import zipfile

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import  pickle
import torch
from dataclasses import dataclass, asdict, replace
# from .common import (
#     SequentialSqueezeAndExcitationTRT,
#     SequentialSqueezeAndExcitation,
#     SqueezeAndExcitation,
#     SqueezeAndExcitationTRT,
# )
from typing import Optional, Callable
import os

import argparse
from functools import partial


@dataclass
class ModelArch:
    pass


@dataclass
class ModelParams:
    def parser(self, name):
        return argparse.ArgumentParser(
            description=f"{name} arguments", add_help=False, usage=""
        )


@dataclass
class OptimizerParams:
    pass


@dataclass
class Model:
    constructor: Callable
    arch: ModelArch
    params: Optional[ModelParams]
    optimizer_params: Optional[OptimizerParams] = None
    checkpoint_url: Optional[str] = None


def torchhub_docstring(name: str):
    return f"""Constructs a {name} model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args:
        pretrained (bool, True): If True, returns a model pretrained on IMAGENET dataset.
    """


class EntryPoint:
    @staticmethod
    def create(name: str, model: Model):
        ep = EntryPoint(name, model)
        ep.__doc__ = torchhub_docstring(name)
        return ep

    def __init__(self, name: str, model: Model):
        self.name = name
        self.model = model

    def __call__(
        self,
        pretrained=True,
        pretrained_from_file=None,
        state_dict_key_map_fn=None,
        **kwargs,
    ):
        assert not (pretrained and (pretrained_from_file is not None))
        params = replace(self.model.params, **kwargs)

        model = self.model.constructor(arch=self.model.arch, **asdict(params))

        state_dict = None
        if pretrained:
            assert self.model.checkpoint_url is not None
            state_dict = torch.hub.load_state_dict_from_url(
                self.model.checkpoint_url,
                map_location=torch.device("cpu"),
                progress=True,
            )

        if pretrained_from_file is not None:
            if os.path.isfile(pretrained_from_file):
                print(
                    "=> loading pretrained weights from '{}'".format(
                        pretrained_from_file
                    )
                )
                state_dict = torch.load(
                    pretrained_from_file, map_location=torch.device("cpu")
                )
            else:
                print(
                    "=> no pretrained weights found at '{}'".format(
                        pretrained_from_file
                    )
                )

        if state_dict is not None:
            state_dict = {
                k[len("module.") :] if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }

            def reshape(t, conv):
                if conv:
                    if len(t.shape) == 4:
                        return t
                    else:
                        return t.view(t.shape[0], -1, 1, 1)
                else:
                    if len(t.shape) == 4:
                        return t.view(t.shape[0], t.shape[1])
                    else:
                        return t

            if state_dict_key_map_fn is not None:
                state_dict = {
                    state_dict_key_map_fn(k): v for k, v in state_dict.items()
                }

            if pretrained and hasattr(model, "ngc_checkpoint_remap"):
                remap_fn = model.ngc_checkpoint_remap(url=self.model.checkpoint_url)
                state_dict = {remap_fn(k): v for k, v in state_dict.items()}

            def _se_layer_uses_conv(m):
                return any(
                    map(
                        partial(isinstance, m),
                        [
                            SqueezeAndExcitationTRT,
                            SequentialSqueezeAndExcitationTRT,
                        ],
                    )
                )

            state_dict = {
                k: reshape(
                    v,
                    conv=_se_layer_uses_conv(
                        dict(model.named_modules())[".".join(k.split(".")[:-2])]
                    ),
                )
                if is_se_weight(k, v)
                else v
                for k, v in state_dict.items()
            }

            model.load_state_dict(state_dict)
        return model

    def parser(self):
        if self.model.params is None:
            return None
        parser = self.model.params.parser(self.name)
        parser.add_argument(
            "--pretrained-from-file",
            default=None,
            type=str,
            metavar="PATH",
            help="load weights from local file",
        )
        if self.model.checkpoint_url is not None:
            parser.add_argument(
                "--pretrained",
                default=False,
                action="store_true",
                help="load pretrained weights from NGC",
            )

        return parser


def is_se_weight(key, value):
    return key.endswith("squeeze.weight") or key.endswith("expand.weight")


def create_entrypoint(m: Model):
    def _ep(**kwargs):
        params = replace(m.params, **kwargs)
        return m.constructor(arch=m.arch, **asdict(params))

    return _ep




import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import torch
import warnings
from torch import nn
import torch.nn.functional as F

try:
    from pytorch_quantization import nn as quant_nn
except ImportError as e:
    warnings.warn(
        "pytorch_quantization module not found, quantization will not be available"
    )
    quant_nn = None


# LayerBuilder {{{
class LayerBuilder(object):
    @dataclass
    class Config:
        activation: str = "relu"
        conv_init: str = "fan_in"
        bn_momentum: Optional[float] = None
        bn_epsilon: Optional[float] = None

    def __init__(self, config: "LayerBuilder.Config"):
        self.config = config

    def conv(
        self,
        kernel_size,
        in_planes,
        out_planes,
        groups=1,
        stride=1,
        bn=False,
        zero_init_bn=False,
        act=False,
    ):
        conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            padding=int((kernel_size - 1) / 2),
            bias=False,
        )

        nn.init.kaiming_normal_(
            conv.weight, mode=self.config.conv_init, nonlinearity="relu"
        )
        layers = [("conv", conv)]
        if bn:
            layers.append(("bn", self.batchnorm(out_planes, zero_init_bn)))
        if act:
            layers.append(("act", self.activation()))

        if bn or act:
            return nn.Sequential(OrderedDict(layers))
        else:
            return conv

    def convDepSep(
        self, kernel_size, in_planes, out_planes, stride=1, bn=False, act=False
    ):
        """3x3 depthwise separable convolution with padding"""
        c = self.conv(
            kernel_size,
            in_planes,
            out_planes,
            groups=in_planes,
            stride=stride,
            bn=bn,
            act=act,
        )
        return c

    def conv3x3(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """3x3 convolution with padding"""
        c = self.conv(
            3, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """1x1 convolution with padding"""
        c = self.conv(
            1, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """7x7 convolution with padding"""
        c = self.conv(
            7, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """5x5 convolution with padding"""
        c = self.conv(
            5, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def batchnorm(self, planes, zero_init=False):
        bn_cfg = {}
        if self.config.bn_momentum is not None:
            bn_cfg["momentum"] = self.config.bn_momentum
        if self.config.bn_epsilon is not None:
            bn_cfg["eps"] = self.config.bn_epsilon

        bn = nn.BatchNorm2d(planes, **bn_cfg)
        gamma_init_val = 0 if zero_init else 1
        nn.init.constant_(bn.weight, gamma_init_val)
        nn.init.constant_(bn.bias, 0)

        return bn

    def activation(self):
        return {
            "silu": lambda: nn.SiLU(inplace=True),
            "relu": lambda: nn.ReLU(inplace=True),
            "onnx-silu": ONNXSiLU,
        }[self.config.activation]()


# LayerBuilder }}}

# LambdaLayer {{{
class LambdaLayer(nn.Module):
    def __init__(self, lmbd):
        super().__init__()
        self.lmbd = lmbd

    def forward(self, x):
        return self.lmbd(x)


# }}}

# SqueezeAndExcitation {{{
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, squeeze, activation):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.Linear(in_channels, squeeze)
        self.expand = nn.Linear(squeeze, in_channels)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self._attention(x)

    def _attention(self, x):
        out = torch.mean(x, [2, 3])
        out = self.squeeze(out)
        out = self.activation(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        return out


class SqueezeAndExcitationTRT(nn.Module):
    def __init__(self, in_channels, squeeze, activation):
        super(SqueezeAndExcitationTRT, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(in_channels, squeeze, 1)
        self.expand = nn.Conv2d(squeeze, in_channels, 1)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self._attention(x)

    def _attention(self, x):
        out = self.pooling(x)
        out = self.squeeze(out)
        out = self.activation(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        return out


# }}}

# EMA {{{
class EMA:
    def __init__(self, mu, module_ema):
        self.mu = mu
        self.module_ema = module_ema

    def __call__(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            mu = min(self.mu, (1.0 + step) / (10 + step))

        def strip_module(s: str) -> str:
            return s

        mesd = self.module_ema.state_dict()
        with torch.no_grad():
            for name, x in module.state_dict().items():
                if name.endswith("num_batches_tracked"):
                    continue
                n = strip_module(name)
                mesd[n].mul_(mu)
                mesd[n].add_((1.0 - mu) * x)


# }}}

# ONNXSiLU {{{
# Since torch.nn.SiLU is not supported in ONNX,
# it is required to use this implementation in exported model (15-20% more GPU memory is needed)
class ONNXSiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ONNXSiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# }}}


class SequentialSqueezeAndExcitation(SqueezeAndExcitation):
    def __init__(self, in_channels, squeeze, activation, quantized=False):
        super().__init__(in_channels, squeeze, activation)
        self.quantized = quantized
        if quantized:
            assert quant_nn is not None, "pytorch_quantization is not available"
            self.mul_a_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
            self.mul_b_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
        else:
            self.mul_a_quantizer = nn.Identity()
            self.mul_b_quantizer = nn.Identity()

    def forward(self, x):
        out = self._attention(x)
        if not self.quantized:
            return out * x
        else:
            x_quant = self.mul_a_quantizer(out)
            return x_quant * self.mul_b_quantizer(x)


class SequentialSqueezeAndExcitationTRT(SqueezeAndExcitationTRT):
    def __init__(self, in_channels, squeeze, activation, quantized=False):
        super().__init__(in_channels, squeeze, activation)
        self.quantized = quantized
        if quantized:
            assert quant_nn is not None, "pytorch_quantization is not available"
            self.mul_a_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
            self.mul_b_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
        else:
            self.mul_a_quantizer = nn.Identity()
            self.mul_b_quantizer = nn.Identity()

    def forward(self, x):
        out = self._attention(x)
        if not self.quantized:
            return out * x
        else:
            x_quant = self.mul_a_quantizer(out)
            return x_quant * self.mul_b_quantizer(x)


class StochasticDepthResidual(nn.Module):
    def __init__(self, survival_prob: float):
        super().__init__()
        self.survival_prob = survival_prob
        self.register_buffer("mask", torch.ones(()), persistent=False)

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return torch.add(residual, other=x)
        else:
            with torch.no_grad():
                mask = F.dropout(
                    self.mask,
                    p=1 - self.survival_prob,
                    training=self.training,
                    inplace=False,
                )
            return torch.addcmul(residual, mask, x)

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1).squeeze(-1)








import argparse
import random
import math
import warnings
from typing import List, Any, Optional
from collections import namedtuple, OrderedDict
from dataclasses import dataclass, replace

import torch
from torch import nn
from functools import partial

try:
    from pytorch_quantization import nn as quant_nn
    from .quantization import switch_on_quantization
except ImportError as e:
    warnings.warn(
        "pytorch_quantization module not found, quantization will not be available"
    )
    quant_nn = None

    import contextlib

    @contextlib.contextmanager
    def switch_on_quantization(do_quantization=False):
        assert not do_quantization, "quantization is not available"
        try:
            yield
        finally:
            pass


# from .common import (
#     SequentialSqueezeAndExcitation,
#     SequentialSqueezeAndExcitationTRT,
#     LayerBuilder,
#     StochasticDepthResidual,
#     Flatten,
# )

# from .model import (
#     Model,
#     ModelParams,
#     ModelArch,
#     OptimizerParams,
#     create_entrypoint,
#     EntryPoint,
# )


# EffNetArch {{{
@dataclass
class EffNetArch(ModelArch):
    block: Any
    stem_channels: int
    feature_channels: int
    kernel: List[int]
    stride: List[int]
    num_repeat: List[int]
    expansion: List[int]
    channels: List[int]
    default_image_size: int
    squeeze_excitation_ratio: float = 0.25

    def enumerate(self):
        return enumerate(
            zip(
                self.kernel, self.stride, self.num_repeat, self.expansion, self.channels
            )
        )

    def num_layers(self):
        _f = lambda l: len(set(map(len, l)))
        l = [self.kernel, self.stride, self.num_repeat, self.expansion, self.channels]
        assert _f(l) == 1
        return len(self.kernel)

    @staticmethod
    def _scale_width(width_coeff, divisor=8):
        def _sw(num_channels):
            num_channels *= width_coeff
            # Rounding should not go down by more than 10%
            rounded_num_channels = max(
                divisor, int(num_channels + divisor / 2) // divisor * divisor
            )
            if rounded_num_channels < 0.9 * num_channels:
                rounded_num_channels += divisor
            return rounded_num_channels

        return _sw

    @staticmethod
    def _scale_depth(depth_coeff):
        def _sd(num_repeat):
            return int(math.ceil(num_repeat * depth_coeff))

        return _sd

    def scale(self, wc, dc, dis, divisor=8) -> "EffNetArch":
        sw = EffNetArch._scale_width(wc, divisor=divisor)
        sd = EffNetArch._scale_depth(dc)

        return EffNetArch(
            block=self.block,
            stem_channels=sw(self.stem_channels),
            feature_channels=sw(self.feature_channels),
            kernel=self.kernel,
            stride=self.stride,
            num_repeat=list(map(sd, self.num_repeat)),
            expansion=self.expansion,
            channels=list(map(sw, self.channels)),
            default_image_size=dis,
            squeeze_excitation_ratio=self.squeeze_excitation_ratio,
        )


# }}}
# EffNetParams {{{
@dataclass
class EffNetParams(ModelParams):
    dropout: float
    num_classes: int = 1000
    activation: str = "silu"
    conv_init: str = "fan_in"
    bn_momentum: float = 1 - 0.99
    bn_epsilon: float = 1e-3
    survival_prob: float = 1
    quantized: bool = False
    trt: bool = False

    def parser(self, name):
        p = super().parser(name)
        p.add_argument(
            "--num_classes",
            metavar="N",
            default=self.num_classes,
            type=int,
            help="number of classes",
        )
        p.add_argument(
            "--conv_init",
            default=self.conv_init,
            choices=["fan_in", "fan_out"],
            type=str,
            help="initialization mode for convolutional layers, see https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_",
        )
        p.add_argument(
            "--bn_momentum",
            default=self.bn_momentum,
            type=float,
            help="Batch Norm momentum",
        )
        p.add_argument(
            "--bn_epsilon",
            default=self.bn_epsilon,
            type=float,
            help="Batch Norm epsilon",
        )
        p.add_argument(
            "--survival_prob",
            default=self.survival_prob,
            type=float,
            help="Survival probability for stochastic depth",
        )
        p.add_argument(
            "--dropout", default=self.dropout, type=float, help="Dropout drop prob"
        )
        p.add_argument("--trt", metavar="True|False", default=self.trt, type=bool)
        return p


# }}}


class EfficientNet(nn.Module):
    def __init__(
        self,
        arch: EffNetArch,
        dropout: float,
        num_classes: int = 6,
        activation: str = "silu",
        conv_init: str = "fan_in",
        bn_momentum: float = 1 - 0.99,
        bn_epsilon: float = 1e-3,
        survival_prob: float = 1,
        quantized: bool = False,
        trt: bool = False,
    ):
        self.quantized = quantized
        with switch_on_quantization(self.quantized):
            super(EfficientNet, self).__init__()
            self.arch = arch
            self.num_layers = arch.num_layers()
            self.num_blocks = sum(arch.num_repeat)
            self.survival_prob = survival_prob
            self.builder = LayerBuilder(
                LayerBuilder.Config(
                    activation=activation,
                    conv_init=conv_init,
                    bn_momentum=bn_momentum,
                    bn_epsilon=bn_epsilon,
                )
            )

            self.stem = self._make_stem(arch.stem_channels)
            out_channels = arch.stem_channels

            plc = 0
            layers = []
            for i, (k, s, r, e, c) in arch.enumerate():
                layer, out_channels = self._make_layer(
                    block=arch.block,
                    kernel_size=k,
                    stride=s,
                    num_repeat=r,
                    expansion=e,
                    in_channels=out_channels,
                    out_channels=c,
                    squeeze_excitation_ratio=arch.squeeze_excitation_ratio,
                    prev_layer_count=plc,
                    trt=trt,
                )
                plc = plc + r
                layers.append(layer)
            self.layers = nn.Sequential(*layers)
            self.features = self._make_features(out_channels, arch.feature_channels)
            self.classifier = self._make_classifier(
                arch.feature_channels, num_classes, dropout
            )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.features(x)
        x = self.classifier(x)

        return x

    def extract_features(self, x, layers=None):
        if layers is None:
            layers = [f"layer{i+1}" for i in range(self.num_layers)] + [
                "features",
                "classifier",
            ]

        run = [
            i
            for i in range(self.num_layers)
            if "classifier" in layers
            or "features" in layers
            or any([f"layer{j+1}" in layers for j in range(i, self.num_layers)])
        ]

        output = {}
        x = self.stem(x)
        for l in run:
            fn = self.layers[l]
            x = fn(x)
            if f"layer{l+1}" in layers:
                output[f"layer{l+1}"] = x

        if "features" in layers or "classifier" in layers:
            x = self.features(x)
            if "features" in layers:
                output["features"] = x

        if "classifier" in layers:
            output["classifier"] = self.classifier(x)

        return output

    # helper functions {{{
    def _make_stem(self, stem_width):
        return nn.Sequential(
            OrderedDict(
                [
                    ("conv", self.builder.conv3x3(3, stem_width, stride=2)),
                    ("bn", self.builder.batchnorm(stem_width)),
                    ("activation", self.builder.activation()),
                ]
            )
        )

    def _get_survival_prob(self, block_id):
        drop_rate = 1.0 - self.survival_prob
        sp = 1.0 - drop_rate * float(block_id) / self.num_blocks
        return sp

    def _make_features(self, in_channels, num_features):
        return nn.Sequential(
            OrderedDict(
                [
                    ("conv", self.builder.conv1x1(in_channels, num_features)),
                    ("bn", self.builder.batchnorm(num_features)),
                    ("activation", self.builder.activation()),
                ]
            )
        )

    def _make_classifier(self, num_features, num_classes, dropout):
        return nn.Sequential(
            OrderedDict(
                [
                    ("pooling", nn.AdaptiveAvgPool2d(1)),
                    ("squeeze", Flatten()),
                    ("dropout", nn.Dropout(dropout)),
                    ("fc", nn.Linear(num_features, num_classes)),
                ]
            )
        )

    def _make_layer(
        self,
        block,
        kernel_size,
        stride,
        num_repeat,
        expansion,
        in_channels,
        out_channels,
        squeeze_excitation_ratio,
        prev_layer_count,
        trt,
    ):
        layers = []

        idx = 0
        survival_prob = self._get_survival_prob(idx + prev_layer_count)
        blk = block(
            self.builder,
            kernel_size,
            in_channels,
            out_channels,
            expansion,
            stride,
            self.arch.squeeze_excitation_ratio,
            survival_prob if stride == 1 and in_channels == out_channels else 1.0,
            self.quantized,
            trt=trt,
        )
        layers.append((f"block{idx}", blk))

        for idx in range(1, num_repeat):
            survival_prob = self._get_survival_prob(idx + prev_layer_count)
            blk = block(
                self.builder,
                kernel_size,
                out_channels,
                out_channels,
                expansion,
                1,  # stride
                squeeze_excitation_ratio,
                survival_prob,
                self.quantized,
                trt=trt,
            )
            layers.append((f"block{idx}", blk))
        return nn.Sequential(OrderedDict(layers)), out_channels

    def ngc_checkpoint_remap(self, url=None, version=None):
        if version is None:
            version = url.split("/")[8]

        def to_sequential_remap(s):
            splited = s.split(".")
            if splited[0].startswith("layer"):
                return ".".join(
                    ["layers." + str(int(splited[0][len("layer") :]) - 1)] + splited[1:]
                )
            else:
                return s

        def no_remap(s):
            return s

        return {"20.12.0": to_sequential_remap, "21.03.0": to_sequential_remap}.get(
            version, no_remap
        )


# }}}

# MBConvBlock {{{
class MBConvBlock(nn.Module):
    __constants__ = ["quantized"]

    def __init__(
        self,
        builder: LayerBuilder,
        depsep_kernel_size: int,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        squeeze_excitation_ratio: float,
        squeeze_hidden=False,
        survival_prob: float = 1.0,
        quantized: bool = False,
        trt: bool = False,
    ):
        super().__init__()
        self.quantized = quantized
        self.residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        squeeze_base = hidden_dim if squeeze_hidden else in_channels
        squeeze_dim = max(1, int(squeeze_base * squeeze_excitation_ratio))

        self.expand = (
            None
            if in_channels == hidden_dim
            else builder.conv1x1(in_channels, hidden_dim, bn=True, act=True)
        )
        self.depsep = builder.convDepSep(
            depsep_kernel_size, hidden_dim, hidden_dim, stride, bn=True, act=True
        )
        if trt or self.quantized:
            # Need TRT mode for quantized in order to automatically insert quantization before pooling
            self.se: nn.Module = SequentialSqueezeAndExcitationTRT(
                hidden_dim, squeeze_dim, builder.activation(), self.quantized
            )
        else:
            self.se: nn.Module = SequentialSqueezeAndExcitation(
                hidden_dim, squeeze_dim, builder.activation(), self.quantized
            )

        self.proj = builder.conv1x1(hidden_dim, out_channels, bn=True)

        if survival_prob == 1.0:
            self.residual_add = torch.add
        else:
            self.residual_add = StochasticDepthResidual(survival_prob=survival_prob)
        if self.quantized and self.residual:
            assert quant_nn is not None, "pytorch_quantization is not available"
            self.residual_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )  # TODO QuantConv2d ?!?
        else:
            self.residual_quantizer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.residual:
            return self.proj(
                self.se(self.depsep(x if self.expand is None else self.expand(x)))
            )

        b = self.proj(
            self.se(self.depsep(x if self.expand is None else self.expand(x)))
        )
        if self.quantized:
            x = self.residual_quantizer(x)

        return self.residual_add(x, b)


def original_mbconv(
    builder: LayerBuilder,
    depsep_kernel_size: int,
    in_channels: int,
    out_channels: int,
    expand_ratio: int,
    stride: int,
    squeeze_excitation_ratio: int,
    survival_prob: float,
    quantized: bool,
    trt: bool,
):
    return MBConvBlock(
        builder,
        depsep_kernel_size,
        in_channels,
        out_channels,
        expand_ratio,
        stride,
        squeeze_excitation_ratio,
        squeeze_hidden=False,
        survival_prob=survival_prob,
        quantized=quantized,
        trt=trt,
    )


def widese_mbconv(
    builder: LayerBuilder,
    depsep_kernel_size: int,
    in_channels: int,
    out_channels: int,
    expand_ratio: int,
    stride: int,
    squeeze_excitation_ratio: int,
    survival_prob: float,
    quantized: bool,
    trt: bool,
):
    return MBConvBlock(
        builder,
        depsep_kernel_size,
        in_channels,
        out_channels,
        expand_ratio,
        stride,
        squeeze_excitation_ratio,
        squeeze_hidden=True,
        survival_prob=survival_prob,
        quantized=quantized,
        trt=trt,
    )


# }}}

# EffNet configs {{{
# fmt: off
effnet_b0_layers = EffNetArch(
    block = original_mbconv,
    stem_channels = 32,
    feature_channels=1280,
    kernel     = [ 3,  3,  5,  3,   5,   5,   3],
    stride     = [ 1,  2,  2,  2,   1,   2,   1],
    num_repeat = [ 1,  2,  2,  3,   3,   4,   1],
    expansion  = [ 1,  6,  6,  6,   6,   6,   6],
    channels   = [16, 24, 40, 80, 112, 192, 320],
    default_image_size=224,
)
effnet_b1_layers=effnet_b0_layers.scale(wc=1,   dc=1.1, dis=240)
effnet_b2_layers=effnet_b0_layers.scale(wc=1.1, dc=1.2, dis=260)
effnet_b3_layers=effnet_b0_layers.scale(wc=1.2, dc=1.4, dis=300)
effnet_b4_layers=effnet_b0_layers.scale(wc=1.4, dc=1.8, dis=380)
effnet_b5_layers=effnet_b0_layers.scale(wc=1.6, dc=2.2, dis=456)
effnet_b6_layers=effnet_b0_layers.scale(wc=1.8, dc=2.6, dis=528)
effnet_b7_layers=effnet_b0_layers.scale(wc=2.0, dc=3.1, dis=600)



urls = {
    "efficientnet-b0": "https://api.ngc.nvidia.com/v2/models/nvidia/efficientnet_b0_pyt_amp/versions/20.12.0/files/nvidia_efficientnet-b0_210412.pth",
    "efficientnet-b4": "https://api.ngc.nvidia.com/v2/models/nvidia/efficientnet_b4_pyt_amp/versions/20.12.0/files/nvidia_efficientnet-b4_210412.pth",
    "efficientnet-widese-b0": "https://api.ngc.nvidia.com/v2/models/nvidia/efficientnet_widese_b0_pyt_amp/versions/20.12.0/files/nvidia_efficientnet-widese-b0_210412.pth",
    "efficientnet-widese-b4": "https://api.ngc.nvidia.com/v2/models/nvidia/efficientnet_widese_b4_pyt_amp/versions/20.12.0/files/nvidia_efficientnet-widese-b4_210412.pth",
    "efficientnet-quant-b0": "https://api.ngc.nvidia.com/v2/models/nvidia/efficientnet_b0_pyt_qat_ckpt_fp32/versions/21.03.0/files/nvidia-efficientnet-quant-b0-130421.pth",
    "efficientnet-quant-b4": "https://api.ngc.nvidia.com/v2/models/nvidia/efficientnet_b4_pyt_qat_ckpt_fp32/versions/21.03.0/files/nvidia-efficientnet-quant-b4-130421.pth",
}

def _m(*args, **kwargs):
    return Model(constructor=EfficientNet, *args, **kwargs)

architectures = {
    "efficientnet-b0": _m(arch=effnet_b0_layers, params=EffNetParams(dropout=0.2), checkpoint_url=urls["efficientnet-b0"]),
    "efficientnet-b1": _m(arch=effnet_b1_layers, params=EffNetParams(dropout=0.2)),
    "efficientnet-b2": _m(arch=effnet_b2_layers, params=EffNetParams(dropout=0.3)),
    "efficientnet-b3": _m(arch=effnet_b3_layers, params=EffNetParams(dropout=0.3)),
    "efficientnet-b4": _m(arch=effnet_b4_layers, params=EffNetParams(dropout=0.4, survival_prob=0.8), checkpoint_url=urls["efficientnet-b4"]),
    "efficientnet-b5": _m(arch=effnet_b5_layers, params=EffNetParams(dropout=0.4)),
    "efficientnet-b6": _m(arch=effnet_b6_layers, params=EffNetParams(dropout=0.5)),
    "efficientnet-b7": _m(arch=effnet_b7_layers, params=EffNetParams(dropout=0.5)),
    "efficientnet-widese-b0": _m(arch=replace(effnet_b0_layers, block=widese_mbconv), params=EffNetParams(dropout=0.2), checkpoint_url=urls["efficientnet-widese-b0"]),
    "efficientnet-widese-b1": _m(arch=replace(effnet_b1_layers, block=widese_mbconv), params=EffNetParams(dropout=0.2)),
    "efficientnet-widese-b2": _m(arch=replace(effnet_b2_layers, block=widese_mbconv), params=EffNetParams(dropout=0.3)),
    "efficientnet-widese-b3": _m(arch=replace(effnet_b3_layers, block=widese_mbconv), params=EffNetParams(dropout=0.3)),
    "efficientnet-widese-b4": _m(arch=replace(effnet_b4_layers, block=widese_mbconv), params=EffNetParams(dropout=0.4, survival_prob=0.8), checkpoint_url=urls["efficientnet-widese-b4"]),
    "efficientnet-widese-b5": _m(arch=replace(effnet_b5_layers, block=widese_mbconv), params=EffNetParams(dropout=0.4)),
    "efficientnet-widese-b6": _m(arch=replace(effnet_b6_layers, block=widese_mbconv), params=EffNetParams(dropout=0.5)),
    "efficientnet-widese-b7": _m(arch=replace(effnet_b7_layers, block=widese_mbconv), params=EffNetParams(dropout=0.5)),
    "efficientnet-quant-b0": _m(arch=effnet_b0_layers, params=EffNetParams(dropout=0.2, quantized=True), checkpoint_url=urls["efficientnet-quant-b0"]),
    "efficientnet-quant-b1": _m(arch=effnet_b1_layers, params=EffNetParams(dropout=0.2, quantized=True)),
    "efficientnet-quant-b2": _m(arch=effnet_b2_layers, params=EffNetParams(dropout=0.3, quantized=True)),
    "efficientnet-quant-b3": _m(arch=effnet_b3_layers, params=EffNetParams(dropout=0.3, quantized=True)),
    "efficientnet-quant-b4": _m(arch=effnet_b4_layers, params=EffNetParams(dropout=0.4, survival_prob=0.8, quantized=True), checkpoint_url=urls["efficientnet-quant-b4"]),
    "efficientnet-quant-b5": _m(arch=effnet_b5_layers, params=EffNetParams(dropout=0.4, quantized=True)),
    "efficientnet-quant-b6": _m(arch=effnet_b6_layers, params=EffNetParams(dropout=0.5, quantized=True)),
    "efficientnet-quant-b7": _m(arch=effnet_b7_layers, params=EffNetParams(dropout=0.5, quantized=True)),
}
# fmt: on

# }}}

_ce = lambda n: EntryPoint.create(n, architectures[n])

# efficientnet_b0 = _ce("efficientnet-b0")
# efficientnet_b4 = _ce("efficientnet-b4")

# efficientnet_widese_b0 = _ce("efficientnet-widese-b0")
efficientnet_widese_b4 = _ce("efficientnet-widese-b4")

# efficientnet_quant_b0 = _ce("efficientnet-quant-b0")
# efficientnet_quant_b4 = _ce("efficientnet-quant-b4")




def nvidia_efficientnet(type='efficientnet-widese-b4', pretrained=True, **kwargs):
    """Constructs a EfficientNet model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args:
        pretrained (bool, True): If True, returns a model pretrained on IMAGENET dataset.
    """

    #from .efficientnet import _ce

    return _ce(type)(pretrained=pretrained, **kwargs)


model  = nvidia_efficientnet()
model.classifier.fc = nn.Linear(in_features=1792,out_features=6)
#model.load_state_dict(torch.load("efficent_b4_Weights.pth"))
model.load_state_dict(torch.load("efficent_b4_WeightsLast.pth"))

#torch.save(model.state_dict(),"efficent_b4_Weights22.pth")
#model = torch.load("efficient96")
#print (model.eval())
