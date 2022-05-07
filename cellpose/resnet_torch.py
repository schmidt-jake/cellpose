import datetime
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torch.nn.parameter import Parameter

from . import dynamics
from . import io
from . import transforms
from . import utils

sz = 3


def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )


def batchconv(in_channels: int, out_channels: int, sz: int) -> Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels: int, out_channels: int, sz: int) -> Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sz: int) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t == 0:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(in_channels, out_channels, sz)
                )
            else:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(out_channels, out_channels, sz)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(in_channels, out_channels, sz)
                )
            else:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(out_channels, out_channels, sz)
                )

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.Module):
    def __init__(self, nbase: List[int], sz: int, residual_on: bool = True) -> None:
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.add_module(
                    "res_down_%d" % n, resdown(nbase[n], nbase[n + 1], sz)
                )
            else:
                self.down.add_module(
                    "conv_down_%d" % n, convdown(nbase[n], nbase[n + 1], sz)
                )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        concatenation: bool = False,
    ) -> None:
        super().__init__()
        self.concatenation = concatenation
        if concatenation:
            self.conv = batchconv(in_channels * 2, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.conv = batchconv(in_channels, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels)

    def forward(
        self,
        style: torch.Tensor,
        x: torch.Tensor,
        mkldnn: bool = False,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class resup(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        concatenation: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", batchconv(in_channels, out_channels, sz))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels,
                out_channels,
                style_channels,
                sz,
                concatenation=concatenation,
            ),
        )
        self.conv.add_module(
            "conv_2", batchconvstyle(out_channels, out_channels, style_channels, sz)
        )
        self.conv.add_module(
            "conv_3", batchconvstyle(out_channels, out_channels, style_channels, sz)
        )
        self.proj = batchconv0(in_channels, out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor,
        mkldnn: bool = False,
    ) -> torch.Tensor:
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](
            style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn
        )
        return x


class convup(nn.Module):
    def __init__(
        self, in_channels, out_channels, style_channels, sz, concatenation=False
    ):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", batchconv(in_channels, out_channels, sz))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels,
                out_channels,
                style_channels,
                sz,
                concatenation=concatenation,
            ),
        )

    def forward(self, x, y, style, mkldnn=False):
        x = self.conv[1](style, self.conv[0](x), y=y)
        return x


class make_style(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5

        return style


class upsample(nn.Module):
    def __init__(
        self,
        nbase: List[int],
        sz: int,
        residual_on: bool = True,
        concatenation: bool = False,
    ) -> None:
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            if residual_on:
                self.up.add_module(
                    "res_up_%d" % (n - 1),
                    resup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation),
                )
            else:
                self.up.add_module(
                    "conv_up_%d" % (n - 1),
                    convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation),
                )

    def forward(
        self, style: torch.Tensor, xd: List[torch.Tensor], mkldnn: bool = False
    ) -> torch.Tensor:
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


class CPnet(nn.Module):
    def __init__(
        self,
        nbase: List[int],
        nout: int,
        sz: int,
        residual_on: bool = True,
        style_on: bool = True,
        concatenation: bool = False,
        mkldnn: bool = False,
        diam_mean: Union[float, Parameter] = 30.0,
    ) -> None:
        super(CPnet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(
            nbaseup, sz, residual_on=residual_on, concatenation=concatenation
        )
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.diam_mean = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.diam_labels = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.style_on = style_on

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0, self.mkldnn)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()
            # T1 = T1.to_dense()
        return T0, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename: str, cpu: bool = False) -> None:
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(
                self.nbase,
                self.nout,
                self.sz,
                self.residual_on,
                self.style_on,
                self.concatenation,
                self.mkldnn,
                self.diam_mean,
            )
            state_dict = torch.load(filename, map_location=torch.device("cpu"))
        self.load_state_dict(
            dict([(name, param) for name, param in state_dict.items()]), strict=False
        )
