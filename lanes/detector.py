"""Deep-learning pipeline: LaneNet segmentation -> per-slot pixels -> RANSAC + joint parallel-lane fit.

Also returns a per-frame confidence used by the auto-labeling tool:
  lane score  = mean softmax probability of the lane's pixels x RANSAC inlier ratio
  frame score = lowest lane score x geometric consistency, where consistency drops when the joint parallel-lane model
                and independent per-lane fits disagree (a sign that one lane is wrong).
"""
import numpy as np
import torch

from .camera import H_SAMPLES
from .data import small
from .geometry import lanes_from_points
from .model import LaneNet, to_tensor

MIN_PIXELS = 15
MIN_ROWS = 6


def load(path):
    net = LaneNet()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()


@torch.no_grad()
def segment(net, images):
    """Full-resolution BGR images -> softmax probabilities (N x 5 x 128 x 256)."""
    x = to_tensor(np.stack([small(im) for im in images]))
    return torch.softmax(net(x), 1).numpy()


def lanes_from_probs(prob, rng=None, joint=True):
    rng = rng if rng is not None else np.random.default_rng(0)
    lab = prob.argmax(0)
    scale = 256 / prob.shape[1]
    points, mean_p = {}, {}
    for s in range(4):
        ys, xs = np.nonzero(lab == s + 1)
        if len(xs) < MIN_PIXELS:
            continue
        # one probability-weighted centre per row: a slanted lane covers many pixels of the same row
        w = prob[s + 1, ys, xs]
        rows = np.unique(ys)
        cx = np.bincount(ys, w * xs)[rows] / np.bincount(ys, w)[rows]
        if len(rows) < MIN_ROWS:
            continue
        points[s] = ((cx + 0.5) * scale, (rows + 0.5) * scale)
        mean_p[s] = float(w.mean())
    lanes, info = lanes_from_points(points, rng, joint=joint, min_pts=MIN_ROWS)
    for s in info:
        info[s]["mean_prob"] = mean_p[s]
    return lanes, info


def frame_confidence(lanes_joint, lanes_indep, info):
    present = [s for s in range(4) if (lanes_joint[s] >= 0).sum() >= 2]
    if not present or 1 not in present or 2 not in present:     # both ego boundaries are required
        return 0.0
    lane_scores = [info[s]["mean_prob"] * info[s]["inlier_ratio"] for s in present]
    dev = 0.0
    for s in present:
        ok = (lanes_joint[s] >= 0) & (lanes_indep[s] >= 0)
        if ok.any():
            dev = max(dev, float(np.abs(lanes_joint[s][ok] - lanes_indep[s][ok]).max()))
    consistency = float(np.clip(1 - dev / 12.0, 0, 1))
    return float(min(lane_scores) * consistency)


def detect(net, images, joint=True, with_confidence=False):
    probs = segment(net, images)
    out = []
    for p in probs:
        lanes, info = lanes_from_probs(p, joint=joint)
        if with_confidence:
            indep, _ = lanes_from_probs(p, joint=False)
            out.append((lanes, frame_confidence(lanes, indep, info)))
        else:
            out.append(lanes)
    return out


__all__ = ["load", "segment", "lanes_from_probs", "frame_confidence", "detect", "H_SAMPLES"]
