import numpy as np
import torch
from torchvision.transforms.functional import resize

from cellpose import dynamics, transforms
from cellpose.plot import dx_to_circ
from cellpose.resnet_torch import CPnet


class MyNet(CPnet):
    def __init__(self, *args, cellprob_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cellprob_threshold = cellprob_threshold

    def forward(self, x: torch.Tensor, rescale: float = 1.0) -> torch.Tensor:
        # x: NHWC
        x = normalize(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = resize(img=x, size=int(rescale * x.size(-1)))
        y, style = super().forward(x)  # NCHW
        # style /= style.square().sum().sqrt()
        cellprob = y[:, 2, :, :]  # NHW
        dP = y[:, :2, :, :]  # CNHW in cellpose but it should be NCHW
        cellprob = cellprob.numpy()
        dP = dP.numpy()
        niter = int(1 / rescale * 200)
        print("niter:", niter)
        masks = [
            torch.from_numpy(
                dynamics.compute_masks(
                    dP=dP[i, ...],
                    cellprob=cellprob[i, ...],
                    niter=niter,
                )[0].astype(np.int16)
            )
            for i in range(x.size(0))
        ]
        return torch.stack(masks)

    # def compute_masks(self, dP, cellprob):
    #     cp_mask = cellprob > self.cellprob_threshold
    #     # p: (2, 120, 120)
    #     # inds: (4964, 2)
    #     p, inds = self.follow_flows(
    #         dP=dP * cp_mask / 5.0,
    #     )

    # def follow_flows(self, dP: torch.Tensor):
    #     p = torch.meshgrid(np.arange(dP.size(1)), np.arange(dP.size(2)), indexing="ij")
    #     p = torch.stack(p, dim=0)
    #     inds = torch.stack(torch.nonzero(torch.abs(dP[0]) > 1e-3)).int().T
    #     rescale = self.diam_mean / self.diam_labels  # FIXME
    #     steps2D(
    #         p=p.numpy(),
    #         dP=dP.numpy(),
    #         inds=inds.numpy(),
    #         niter=1 / rescale * 200,
    #     )
    #     return p, inds

    # def get_masks(self, p: torch.Tensor, iscell: torch.Tensor):
    #     pflows = []
    #     edges = []
    #     shape0 = p.shape[1:]
    #     dims = len(p)
    #     inds = torch.meshgrid(
    #         torch.arange(shape0[0]), torch.arange(shape0[1]), indexing="ij"
    #     )
    #     for i in range(dims):
    #         p[i, ~iscell] = inds[i][~iscell]


def normalize(x: torch.Tensor) -> torch.Tensor:
    # x is (NHWC)
    x = x.clone()
    x = x.float()
    maximum = x.amax(dim=[1, 2], keepdim=True)
    minimum = x.amin(dim=[1, 2], keepdim=True)
    x.sub_(minimum).div_(maximum - minimum)
    return x


def tile(x: torch.Tensor, size: int, stride: int) -> torch.Tensor:
    # x is (NHWC)
    tiles = x.unfold(2, size, stride).unfold(1, size, stride)
    # tiles is N x tiles_per_img x size x size x C
    return tiles


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


class MyCellposeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nbase = [2, 32, 64, 128, 256]
        self.rescale = 1.0
        self.nclasses = 3
        self.return_conv = False
        self.bsize = 224
        self.augment = False
        self.tile_overlap = 0.1
        self.batch_size = 8

    def _run_nets(self):
        pass

    def _run_net(self, img: torch.Tensor):
        img, ysub, xsub = transforms.pad_image_ND(img)
        y, style = self._run_tiled(imgi=img)
        style /= (style**2).sum() ** 0.5
        slc = self._get_slices(imgs=img, xsub=xsub, ysub=ysub)
        # slice out padding
        y = y[slc]
        # transpose so channels axis is last again
        y = y.permute(1, 2, 0)
        return y, style

    def _run_tiled(self, imgi: torch.Tensor):
        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(
            imgi,
            bsize=self.bsize,
            augment=self.augment,
            tile_overlap=self.tile_overlap,
        )
        ny, nx, nchan, ly, lx = IMG.shape
        IMG = IMG.reshape(ny * nx, nchan, ly, lx)
        batch_size = self.batch_size
        niter = int(np.ceil(IMG.shape[0] / batch_size))
        nout = self.nclasses + 32 * self.return_conv
        y = torch.zeros((IMG.shape[0], nout, ly, lx))
        for k in range(niter):
            irange = np.arange(
                batch_size * k, min(IMG.shape[0], batch_size * k + batch_size)
            )
            y0, style = self.network(IMG[irange], return_conv=self.return_conv)
            y[irange] = y0.reshape(
                len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1]
            )
            if k == 0:
                styles = style[0]
            styles += style.sum(axis=0)
        styles /= IMG.shape[0]
        yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        yf = yf[:, : imgi.shape[1], : imgi.shape[2]]
        styles /= (styles**2).sum() ** 0.5
        return yf, styles

    def _get_slices(self, imgs: torch.Tensor, xsub, ysub):
        slc = [slice(0, imgs.shape[n] + 1) for n in range(imgs.ndim)]
        slc[-3] = slice(0, self.nclasses + 32 * self.return_conv + 1)
        slc[-2] = slice(ysub[0], ysub[-1] + 1)
        slc[-1] = slice(xsub[0], xsub[-1] + 1)
        slc = tuple(slc)  # type: ignore[assignment]
        return slc

    def _run_cp(self, x: torch.Tensor):
        # equivalent to CellposeModel._run_cp
        shape = x.shape
        nimg = shape[0]
        styles = torch.zeros((nimg, self.nbase[-1]), dtype=torch.float32)
        dP = torch.zeros(
            (2, nimg, int(shape[1] * self.rescale), int(shape[2] * self.rescale)),
            dtype=torch.float32,
        )
        cellprob = torch.zeros(
            (nimg, int(shape[1] * self.rescale), int(shape[2] * self.rescale)),
            dtype=torch.float32,
        )
        for i in range(nimg):
            img = x[i]
            img = transforms.normalize_img(img=img)
            yf, style = self._run_net(img=x)
            cellprob[i] = yf[:, :, 2]
            dP[:, i] = yf[:, :, :2].permute((2, 0, 1))
            cellprob[i] = yf[:, :, 2]
            dP[:, i] = yf[:, :, :2].permute((2, 0, 1))
            styles[i] = style
        del yf, style
        styles = styles.squeeze()

        # pass back zeros if not compute_masks
        masks, p = torch.zeros(0), torch.zeros(0)
        return masks, styles, dP, cellprob, p

    def get_styles(self, x: torch.Tensor):
        shape = x.shape
        nimg = shape[0]
        styles = torch.zeros((nimg, self.nbase[-1]), dtype=torch.float32)
        for i in range(nimg):
            img = x[i]
            img = transforms.normalize_img(img=img)
            _, style = self._run_tiled(imgi=img)
            style /= (style**2).sum() ** 0.5
            styles[i] = style
        styles = styles.squeeze()
        return styles

    def forward(self, x: torch.Tensor):
        # equivalent to CellposeModel.eval
        masks, styles, dP, cellprob, p = self._run_cp(x=x)
        flows = [dx_to_circ(dP), dP, cellprob, p]
        return masks, flows, styles

    def get_masks(self):
        pass


class MyCellpose(torch.nn.Module):
    def __init__(self, size_model: SizeModel, cp_model: MyCellposeModel) -> None:
        super().__init__()
        self.diam_mean = 30.0
        self.size_model = size_model
        self.cp_model = cp_model

    def forward(self, x: torch.Tensor):
        masks = self.cp_model.get_masks(x)
        diams = self.size_model(masks=masks)
        rescale = self.diam_mean / np.array(diams)
        return masks, flows, styles, diams
