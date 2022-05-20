from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def convbatchrelu(in_channels: int, out_channels: int, sz: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
        torch.nn.BatchNorm2d(out_channels, eps=1e-5),
        torch.nn.ReLU(inplace=True),
    )


def batchconv(in_channels: int, out_channels: int, sz: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels, eps=1e-5),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels: int, out_channels: int, sz: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels, eps=1e-5),
        torch.nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sz: int) -> None:
        super().__init__()
        self.proj = batchconv0(in_channels, out_channels, 1)
        self.conv = torch.nn.ModuleList(
            [
                batchconv(in_channels, out_channels, sz)
                if t == 0
                else batchconv(out_channels, out_channels, sz)
                for t in range(4)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (4, 2, 224, 224)
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sz: int) -> None:
        super().__init__()
        self.conv0 = batchconv(in_channels, out_channels, sz)
        self.conv1 = batchconv(out_channels, out_channels, sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        return x


class downsample(torch.nn.Module):
    def __init__(self, nbase: List[int], sz: int, residual_on: bool = True) -> None:
        super().__init__()
        self.down = torch.nn.ModuleList()
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.append(resdown(nbase[n], nbase[n + 1], sz))
            else:
                self.down.append(convdown(nbase[n], nbase[n + 1], sz))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        xd: List[torch.Tensor] = []
        for n, down in enumerate(self.down):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(down(y))
        return xd


class batchconvstyle(torch.nn.Module):
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
            self.full = torch.nn.Linear(style_channels, out_channels * 2)
        else:
            self.conv = batchconv(in_channels, out_channels, sz)
            self.full = torch.nn.Linear(style_channels, out_channels)

    def forward(
        self,
        style: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat: torch.Tensor = self.full(style)
        y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class resup(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        concatenation: bool = False,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential()
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
    ) -> torch.Tensor:
        x = self.proj(x) + self.conv[1](style=style, x=self.conv[0](x), y=y)
        x = x + self.conv[3](style=style, x=self.conv[2](style=style, x=x))
        return x


class convup(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        concatenation: bool = False,
    ):
        super().__init__()
        self.conv = torch.nn.Sequential()
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

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv[1](style, self.conv[0](x), y=y)
        return x


class make_style(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.pool_all = torch.nn.AvgPool2d(28)
        self.flatten = torch.nn.Flatten()

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style: torch.Tensor = self.flatten(style)  # type: ignore[no-redef]
        style /= style.square().sum(dim=1, keepdim=True).sqrt()
        # style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5

        return style


class upsample(torch.nn.Module):
    def __init__(
        self,
        nbase: List[int],
        sz: int,
        residual_on: bool = True,
        concatenation: bool = False,
    ) -> None:
        super().__init__()
        self.upsampling = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.up = torch.nn.ModuleList()
        for n in range(1, len(nbase)):
            if residual_on:
                self.up.append(
                    resup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation)
                )
            else:
                self.up.append(
                    convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation)
                )

    def forward(self, style: torch.Tensor, xd: List[torch.Tensor]) -> torch.Tensor:
        # FIXME: make this dynamic
        x = self.up[-1](x=xd[-1], y=xd[-1], style=style)
        x = self.upsampling(x)
        x = self.up[2](x=x, y=xd[2], style=style)
        x = self.upsampling(x)
        x = self.up[1](x=x, y=xd[1], style=style)
        x = self.upsampling(x)
        x = self.up[0](x=x, y=xd[0], style=style)
        return x


class CPnet(torch.nn.Module):
    def __init__(
        self,
        nbase: List[int],
        nout: int,
        sz: int,
        residual_on: bool = True,
        style_on: bool = True,
        concatenation: bool = False,
        diam_mean: Union[float, torch.nn.Parameter] = 30.0,
    ) -> None:
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(
            nbaseup, sz, residual_on=residual_on, concatenation=concatenation
        )
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.diam_mean = torch.nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.diam_labels = torch.nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.style_on = style_on

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # data.shape = [4, 2, 224, 224]
        T0 = self.downsample(data)
        style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0)
        T0 = self.output(T0)
        return T0, style0

    def save_model(self, filename: str) -> None:
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


if __name__ == "__main__":
    net = CPnet(nbase=[2, 32, 64, 128, 256], nout=3, sz=3, concatenation=True)
    net.eval()
    with torch.inference_mode():
        net(torch.rand(size=(1, 2, 256, 256)))
