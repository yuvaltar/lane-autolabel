import os

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

from lanes.data import make_train
from lanes.model import IN_H, IN_W, N_CLASSES, LaneNet, to_tensor

CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "lanenet.pt")


def test_output_shape():
    net = LaneNet().eval()
    assert net(torch.zeros(2, 3, IN_H, IN_W)).shape == (2, N_CLASSES, IN_H, IN_W)


def test_overfits_a_small_batch():
    torch.manual_seed(0)
    X, Y = make_train(range(4), aug=False)
    net = LaneNet(); opt = torch.optim.Adam(net.parameters(), 3e-3)
    x, y = to_tensor(X), torch.from_numpy(Y).long()
    w = torch.tensor([0.2, 1, 1, 1, 1.0])
    first = None
    for _ in range(40):
        loss = Fn.cross_entropy(net(x), y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < 0.3 * first


@pytest.mark.skipif(not os.path.exists(CKPT), reason="no trained checkpoint")
def test_trained_model_regression():
    from lanes import detector
    from lanes.metrics import frame_scores, summarize
    from lanes.synth import render
    net = detector.load(CKPT)
    scenes = [render(s) for s in range(200_000, 200_040)]
    preds = detector.detect(net, [sc.image for sc in scenes])
    assert summarize([frame_scores(p, sc.lanes) for p, sc in zip(preds, scenes)])["accuracy"] > 0.95
