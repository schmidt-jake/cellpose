from typing import Any, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import pad
from torch.types import Number
from torchvision.transforms.functional import resize

from cellpose.dynamics import compute_masks
from cellpose.plot import dx_to_circ, show_segmentation
from cellpose.resnet_torch import CPnet
from cellpose.utils import diameters


def normalize_img(x: torch.Tensor) -> torch.Tensor:
    # x is WHC
    x = x.clone()
    x = x.float()
    for dim in range(x.shape[2]):
        c: torch.Tensor = x[:, :, dim]
        q = c.flatten().quantile(
            q=torch.tensor(
                [0.01, 0.99],
                dtype=x.dtype,
                device=x.device,
            )
        )
        c = (c - q[0]) / (q[1] - q[0])
        x[:, :, dim] = c
    return x


def pad_img(
    img0: torch.Tensor,
    div: int = 16,
    extra: int = 1,
):
    Lpad = int(div * np.ceil(img0.size(-2) / div) - img0.size(-2))
    xpad1 = extra * div // 2 + Lpad // 2
    xpad2 = extra * div // 2 + Lpad - Lpad // 2
    Lpad = int(div * np.ceil(img0.size(-1) / div) - img0.size(-1))
    ypad1 = extra * div // 2 + Lpad // 2
    ypad2 = extra * div // 2 + Lpad - Lpad // 2

    I = pad(img0, [ypad1, ypad2, xpad1, xpad2], mode="constant")

    Ly, Lx = img0.shape[-2:]
    ysub = torch.arange(xpad1, xpad1 + Ly)
    xsub = torch.arange(ypad1, ypad1 + Lx)

    slc = get_slice(I=I, xsub=xsub, ysub=ysub)

    return I, slc


def get_slice(I: torch.Tensor, ysub: torch.Tensor, xsub: torch.Tensor) -> Tuple:
    slc = [slice(0, I.shape[n] + 1) for n in range(I.ndim)]
    slc[-3] = slice(0, 3 + 32)
    slc[-2] = slice(ysub[0], ysub[-1] + 1)
    slc[-1] = slice(xsub[0], xsub[-1] + 1)
    slc = tuple(slc)
    return slc


def resize_img(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    # channels first
    new_size = int(x.size(1) * scale_factor)
    x = resize(img=x, size=[new_size] * 2)
    return x


class SizeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, masks: torch.Tensor) -> int:
        # copied from cellpose.utils.diameters
        _, counts = masks.int().unique(dim=1, return_counts=True)
        counts = counts[:, 1:]
        md = torch.median(counts.sqrt(), dim=1)
        if torch.isnan(md):
            md = 0
        md /= (torch.pi**0.5) / 2
        return md

    @classmethod
    def make_size_model(
        cls: Type["SizeModel"], pretrained_filepath: str
    ) -> "SizeModel":
        params: Dict[str, Any] = np.load(pretrained_filepath, allow_pickle=True).item()
        return cls(
            **{
                k: torch.from_numpy(v).float()
                if isinstance(v, np.ndarray)
                else float(v)
                for k, v in params.items()
            }
        )


class Net(torch.nn.Module):
    def __init__(self, net: CPnet) -> None:
        super().__init__()
        self.net = net

    def forward(self, img: torch.Tensor, resample: bool, scale_factor: float = 1.0):
        x = normalize_img(x=img)
        x = x.permute(2, 0, 1)
        if scale_factor != 1.0:
            x = resize_img(x=x, scale_factor=scale_factor)
        # equivalent to UnetModel._run_net
        x, slc = pad_img(x)
        x = x.unsqueeze(0)
        print(
            "our imgs:",
            x.shape,
            x.dtype,
            x.min().item(),
            x.mean().item(),
            x.max().item(),
        )
        y, style = self.net(x)
        print(
            "our styles:",
            style.shape,
            style.dtype,
            style.min().item(),
            style.mean().item(),
            style.max().item(),
        )
        y = y.squeeze(0)
        style = style.squeeze(0)
        y = y[slc]
        print(
            "our styles:",
            style.shape,
            style.dtype,
            style.min().item(),
            style.mean().item(),
            style.max().item(),
        )
        if resample:
            y = resize_img(x=y, scale_factor=img.size(1) / y.size(1))
        y = y.permute(1, 2, 0)
        return y, style


class Model(torch.nn.Module):
    def __init__(self, size_model: SizeModel, net: Net) -> None:
        super().__init__()
        self.size_model = size_model
        self.net = net

    @staticmethod
    def _compute_mask(dP, cellprob, niter: int, resize: Optional[List[int]] = None):
        mask, p = compute_masks(
            dP=dP,
            cellprob=cellprob,
            niter=niter,
            resize=resize,
        )
        mask = mask.astype(np.int16)
        flows = [dx_to_circ(dP), dP, cellprob, p]
        return torch.from_numpy(mask), flows

    def compute_masks(
        self, y: torch.Tensor, niter: int, resize: Optional[List[int]] = None
    ):
        cellprob = y[..., 2]
        print(
            "our cellprob:",
            cellprob.shape,
            cellprob.dtype,
            cellprob.min().item(),
            cellprob.mean().item(),
            cellprob.max().item(),
        )
        dP = y[..., :2].permute(2, 0, 1)
        print(
            "our dP:",
            dP.shape,
            dP.dtype,
            dP.min().item(),
            dP.mean().item(),
            dP.max().item(),
        )
        masks, flows = self._compute_mask(
            dP=dP.numpy(),
            cellprob=cellprob.numpy(),
            niter=niter,
            resize=resize,
        )
        print(
            "our masks:",
            masks.shape,
            masks.dtype,
            masks.min().item(),
            masks.max().item(),
        )
        return masks, flows

    def compute_diam(self, img: torch.Tensor) -> Number:
        # equivalent to SizeModel getting styles
        y, style = self.net(img.clone(), resample=False, scale_factor=1.0)
        diam_style = self.size_model.estimate_size(style)

        scale_factor = (self.size_model.diam_mean / diam_style).item()

        print("rescale", scale_factor)

        # equivalent to SizeModel getting masks
        print("COMPUTING STYLE FOR DIAMETER......................")
        y, style = self.net(img, resample=False, scale_factor=scale_factor)
        print(
            "our styles:",
            style.shape,
            style.dtype,
            style.min().item(),
            style.mean().item(),
            style.max().item(),
        )
        masks, flows = self.compute_masks(
            y=y, niter=int(1 / scale_factor * 200), resize=[img.size(1)] * 2
        )
        diam = self.size_model.compute_diam(masks)
        print("END......................")
        return diam

    def forward(self, img: torch.Tensor):
        img = img[:, :, [2, 0]]
        diam = self.compute_diam(img)
        print("my diam", diam)
        scale_factor = self.size_model.diam_mean / diam
        y, style = self.net(img, resample=True, scale_factor=scale_factor)
        masks, flows = self.compute_masks(y=y, niter=int(1 / scale_factor * 200))
        return masks, flows


if __name__ == "__main__":
    from cellpose.io import imread
    from cellpose.models import Cellpose

    x = imread("images/channels_012.png")
    x = x[:512, :512, :]

    their_model = Cellpose(model_type="cyto2")
    their_model.cp.mkldnn = False

    their_model.cp.net.load_model(their_model.cp.pretrained_model[0], cpu=True)

    our_size_model = SizeModel.make_size_model(
        pretrained_filepath=their_model.pretrained_size
    )
    our_model = Model(
        size_model=our_size_model.eval(),
        net=Net(their_model.cp.net.eval()),
    ).eval()

    with torch.inference_mode():
        masks, flows, styles, diams = their_model.eval(
            x=x.copy(),
            channels=[3, 1],
            channel_axis=2,
            diameter=None,
            net_avg=False,
            tile=False,
            resample=True,
        )
    show_segmentation(
        fig=plt.figure(figsize=(12, 5)),
        img=x.copy(),
        maski=masks,
        flowi=flows[0],
        file_name="theirs",
    )

    with torch.inference_mode():
        masks, flows = our_model(torch.from_numpy(x.copy()))
    show_segmentation(
        fig=plt.figure(figsize=(12, 5)),
        img=x.copy(),
        maski=masks.numpy(),
        flowi=flows[0],
        file_name="ours",
    )
