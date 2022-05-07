from typing import Tuple

import torch

from cellpose.resnet_torch import CPnet


def normalize_img(img: torch.Tensor) -> torch.Tensor:
    percentiles = torch.quantile(
        img.reshape(2, -1),
        q=torch.tensor([0.01, 0.99]),
        dim=1,
    )
    img = (img - percentiles[0]) / (percentiles[1] - percentiles[0])
    return img


class SizeModel(torch.nn.Module):
    def __init__(self, A, smean, ymean, diam_mean) -> None:
        super().__init__()
        self.A = A
        self.smean = smean
        self.ymean = ymean
        self.diam_mean = diam_mean

    def forward(self, styles: torch.Tensor, masks: torch.Tensor):
        diam_style = self._size_estimation(styles)
        diam = self._diameters(masks)
        return diam, diam_style

    def _size_estimation(self, style: torch.Tensor) -> torch.Tensor:
        """linear regression from style to size

        sizes were estimated using "diameters" from square estimates not circles;
        therefore a conversion factor is included (to be removed)

        """
        szest = torch.exp(
            self.A @ (style - self.smean).T + torch.log(self.diam_mean) + self.ymean
        )
        szest = torch.maximum(torch.tensor(5.0), szest)
        return szest

    def _diameters(self, masks: torch.Tensor) -> Tuple:
        _, counts = torch.unique(masks, return_counts=True)
        counts = counts[1:]
        md = torch.median(counts**0.5)
        if torch.isnan(md):
            md = 0
        md /= (torch.pi**0.5) / 2
        return md, counts**0.5


class CellposeModel(torch.nn.Module):
    def __init__(self, net: CPnet) -> None:
        super().__init__()
        self.net = net

    def _run_nets():
        y, style = self._run_tiled(
            imgs,
            augment=augment,
            bsize=bsize,
            tile_overlap=tile_overlap,
            return_conv=return_conv,
        )

    def forward(self, img: torch.Tensor):
        shape = img.shape
        dP = torch.zeros((2, shape[0], shape[1], shape[2]), dtype=torch.float32)
        cellprob = torch.zeros((shape[0], shape[1], shape[2]), dtype=torch.float32)
        img = normalize_img(img)
        yf, style = self._run_nets()  # TODO
        yf = resize_image(yf, shape[1], shape[2])
        cellprob[i] = yf[:, :, 2]
        dP[:, i] = yf[:, :, :2].transpose((2, 0, 1))
        if self.nclasses == 4:
            if i == 0:
                bd = np.zeros_like(cellprob)
            bd[i] = yf[:, :, 3]
        styles[i] = style


class Cellpose(torch.nn.Module):
    def __init__(self, sz: SizeModel, cp: CellposeModel) -> None:
        super().__init__()
        self.sz = sz
        self.cp = cp

    def forward(self, x: torch.Tensor):
        styles = self.cp.styles(x)
        masks = self.cp.masks(x)
        diams, _ = self.size_model(styles=styles, masks=masks)


if __name__ == "__main__":
    img = torch.randint(
        low=0,
        high=255,
        size=(512, 512, 2),
        dtype=torch.uint8,
    )
    print(normalize_img(img.float()))
