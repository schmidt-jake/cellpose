from typing import Tuple

import numpy.typing as npt
import torch
from cellpose import modelsV2
from cellpose.io import imread
from cellpose.models import Cellpose
from cellpose.utils import outlines_list


def test_normalize():
    x = torch.randint(
        low=0,
        high=255,
        size=(3, 512, 512, 2),
        dtype=torch.uint8,
    )
    normed: torch.Tensor = modelsV2.normalize(x=x)
    assert normed.amin(dim=(1, 2, 3)).eq(0.0).all().item()
    assert normed.amax(dim=(1, 2, 3)).eq(1.0).all().item()
    assert normed.shape == x.shape


def test_tile():
    x = torch.rand(size=(3, 512, 512, 2))
    tiled = modelsV2.tile(x=x, size=32, stride=29)
    print(tiled.shape)


def test_my_net():
    x = imread("images/channels_012.png")
    x = torch.from_numpy(x)
    x = x[:256, :256, [2, 0]]
    x = x.unsqueeze(0)
    print(x.shape, x.dtype)
    net = modelsV2.MyNet(
        nbase=[2, 32, 64, 128, 256],
        nout=3,
        sz=3,
    )
    net.load_model(
        "/Users/jacob.schmidt/.cellpose/models/cyto2torch_0",
        cpu=True,
    )
    net.eval()
    with torch.inference_mode():
        torch.jit.trace(net, example_inputs=x)
    with torch.inference_mode():
        # torch.jit.script(net, example_inputs=[(x,)])
        masks: torch.Tensor = net(x)
        masks = masks.squeeze(dim=0).numpy()
    print("masks:", masks.shape, masks.dtype, masks.min(), masks.max())
    print(len(outlines_list(masks)))
    # size_model = modelsV2.SizeModel()
    # diam = size_model(masks=masks)


def test_model() -> None:
    model = Cellpose(model_type="cyto2", net_avg=False)

    print(model.cp.pretrained_model)

    img = imread("images/channels_012.png")

    img: npt.NDArray = img[:256, :256, :]

    # img = torch.from_numpy(img)

    masks, flows, styles, diams = model.eval(
        x=img,
        channels=[3, 1],
        channel_axis=2,
        diameter=30.0,
        net_avg=False,
        tile=False,
        resample=False,
    )
