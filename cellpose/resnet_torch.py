from typing import List, Optional, Tuple

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


class resdown(torch.jit.ScriptModule):
    def __init__(self, in_channels: int, out_channels: int, sz: int):
        super().__init__()
        self.proj = batchconv0(in_channels=in_channels, out_channels=out_channels, sz=1)
        self.conv0 = batchconv(
            in_channels=in_channels, out_channels=out_channels, sz=sz
        )
        self.conv1 = batchconv(
            in_channels=out_channels, out_channels=out_channels, sz=sz
        )
        self.conv2 = batchconv(
            in_channels=out_channels, out_channels=out_channels, sz=sz
        )
        self.conv3 = batchconv(
            in_channels=out_channels, out_channels=out_channels, sz=sz
        )

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.conv1(self.conv0(x))
        x += self.conv3(self.conv2(x))
        return x


class convdown(torch.jit.ScriptModule):
    def __init__(self, in_channels: int, out_channels: int, sz: int):
        super().__init__()
        self.conv0 = batchconv(
            in_channels=in_channels, out_channels=out_channels, sz=sz
        )
        self.conv1 = batchconv(
            in_channels=out_channels, out_channels=out_channels, sz=sz
        )

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        return x


class downsample(torch.jit.ScriptModule):
    def __init__(self, nbase: List[int], sz: int, residual_on: bool = True):
        super().__init__()
        self.down = torch.nn.ModuleList()
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.append(resdown(nbase[n], nbase[n + 1], sz))
            else:
                self.down.append(convdown(nbase[n], nbase[n + 1], sz))

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        xd: List[torch.Tensor] = []
        for n, m in enumerate(self.down):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(m(x=y))
        return xd


class batchconvstyle(torch.jit.ScriptModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        concatenation: bool = False,
    ):
        super().__init__()
        self.concatenation = concatenation
        if concatenation:
            self.conv = batchconv(
                in_channels=in_channels * 2, out_channels=out_channels, sz=sz
            )
            self.full = torch.nn.Linear(
                in_features=style_channels, out_features=out_channels * 2
            )
        else:
            self.conv = batchconv(
                in_channels=in_channels, out_channels=out_channels, sz=sz
            )
            self.full = torch.nn.Linear(
                in_features=style_channels, out_features=out_channels
            )

    @torch.jit.script_method
    def forward(
        self, style: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat: torch.Tensor = self.full(style)
        y_out = x + feat.unsqueeze(-1).unsqueeze(-1)
        y_out = self.conv(y_out)
        return y_out


class resup(torch.jit.ScriptModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        concatenation: bool = False,
    ):
        super().__init__()
        self.conv0 = batchconv(
            in_channels=in_channels, out_channels=out_channels, sz=sz
        )
        self.conv1 = batchconvstyle(
            in_channels=out_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            sz=sz,
            concatenation=concatenation,
        )
        self.conv2 = batchconvstyle(
            in_channels=out_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            sz=sz,
        )
        self.conv3 = batchconvstyle(
            in_channels=out_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            sz=sz,
        )
        self.proj = batchconv0(in_channels=in_channels, out_channels=out_channels, sz=1)

    @torch.jit.script_method
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, style: torch.Tensor
    ) -> torch.Tensor:
        x = self.proj(x) + self.conv1(style=style, x=self.conv0(x), y=y)
        x = x + self.conv3(style=style, x=self.conv2(style=style, x=x))
        return x


class convup(torch.jit.ScriptModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        sz: int,
        concatenation: bool = False,
    ):
        super().__init__()
        self.conv0 = batchconv(in_channels, out_channels, sz)
        self.conv1 = batchconvstyle(
            in_channels=out_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            sz=sz,
            concatenation=concatenation,
        )

    @torch.jit.script_method
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, style: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(style=style, x=self.conv0(x), y=y)
        return x


class make_style(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        # self.pool_all = torch.nn.AvgPool2d(28)
        self.flatten = torch.nn.Flatten()

    @torch.jit.script_method
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style: torch.Tensor = self.flatten(style)  # type: ignore[no-redef]
        style /= style.square().sum(dim=1, keepdim=True).sqrt()
        # style /= torch.sum(style**2, axis=1, keepdim=True) ** 0.5
        return style


class upsample(torch.jit.ScriptModule):
    def __init__(
        self,
        nbase: List[int],
        sz: int,
        residual_on: bool = True,
        concatenation: bool = False,
    ):
        super().__init__()
        self.upsampling = torch.nn.Upsample(scale_factor=2, mode="nearest")
        _up: List[torch.nn.Module] = []
        for n in range(1, len(nbase)):
            if residual_on:
                _up.append(
                    resup(
                        in_channels=nbase[n],
                        out_channels=nbase[n - 1],
                        style_channels=nbase[-1],
                        sz=sz,
                        concatenation=concatenation,
                    )
                )
            else:
                _up.append(
                    convup(
                        in_channels=nbase[n],
                        out_channels=nbase[n - 1],
                        style_channels=nbase[-1],
                        sz=sz,
                        concatenation=concatenation,
                    )
                )
        _up.reverse()
        self.up = torch.nn.ModuleList(_up)

    @torch.jit.script_method
    def forward(self, style: torch.Tensor, xd: List[torch.Tensor]) -> torch.Tensor:
        xd = xd[::-1]
        x = self.up[0](x=xd[0], y=xd[0], style=style)
        for n, m in enumerate(self.up[1:]):
            x = self.upsampling(x)
            x = m.forward(x=x, y=xd[n], style=style)
        return x


class CPnet(torch.jit.ScriptModule):
    def __init__(
        self,
        nbase: List[int],
        nout: int,
        sz: int,
        residual_on: bool = True,
        style_on: bool = True,
        concatenation: bool = False,
        diam_mean: float = 30.0,
    ):
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.downsample = downsample(nbase=nbase, sz=sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(
            nbase=nbaseup, sz=sz, residual_on=residual_on, concatenation=concatenation
        )
        self.make_style = make_style()
        self.output = batchconv(in_channels=nbaseup[0], out_channels=nout, sz=1)
        self.diam_mean = torch.nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.diam_labels = torch.nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.style_on = style_on

    @torch.jit.script_method
    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
