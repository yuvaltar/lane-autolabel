"""Synthetic road-scene generator with exact lane ground truth.

Every scene is fully determined by its integer seed, so a dataset is just a seed range and train/val/test never overlap.
Randomized per scene: lane width, ego offset and heading, road curvature, number of lanes, marking type (solid /
dashed, white / yellow) and wear, view distance, camera pitch, asphalt texture, cracks and repair patches, vehicles
that occlude the markings, and the background.

Lane slots (class ids): 1 = second boundary on the left, 2 = ego-left, 3 = ego-right, 4 = second on the right.
Ground truth per slot: the lane's column at every row of H_SAMPLES, or -2 where the lane is not visible (TuSimple format).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import F, H, H_CAM, H_SAMPLES, U0, V0, W, project, row_to_z

N_SLOTS = 4
SLOT_OFFSETS = np.array([-1.5, -0.5, 0.5, 1.5])      # boundary position, in lane widths from the ego-lane centre
WHITE = np.array([235, 235, 230], float)
YELLOW = np.array([40, 190, 230], float)            # BGR


@dataclass
class Scene:
    image: np.ndarray                 # H x W x 3, uint8, BGR
    lanes: np.ndarray                 # N_SLOTS x len(H_SAMPLES), column or -2
    present: np.ndarray               # N_SLOTS bool
    v0: float                         # true horizon row of this scene
    params: dict = field(default_factory=dict)


def lane_x(params, slot, Z):
    """Lateral position (m) of boundary `slot` (0-based) at distance Z."""
    p = params
    return p["w"] * SLOT_OFFSETS[slot] - p["d"] + p["t"] * Z + 0.5 * p["c"] * Z ** 2


def _sample_params(rng):
    p = dict(
        w=rng.uniform(3.2, 3.9),               # lane width
        d=rng.uniform(-0.5, 0.5),              # ego offset from lane centre
        t=rng.uniform(-0.03, 0.03),            # heading (rad, small-angle)
        c=rng.uniform(-1.6e-3, 1.6e-3),        # curvature (1/m)
        zmax=rng.uniform(45, 85),              # how far the road is visible
        v0=V0 + rng.uniform(-3, 3),            # pitch jitter
    )
    present = np.array([rng.random() < 0.75, True, True, rng.random() < 0.75])
    style = []
    for s in range(N_SLOTS):
        outer = s in (0, 3)
        dashed = rng.random() < (0.35 if outer else 0.6)
        yellow = (s == 1 and rng.random() < 0.25) or (outer and rng.random() < 0.1)
        style.append(dict(dashed=dashed, yellow=yellow, width=rng.uniform(0.10, 0.20),
                          dash=rng.uniform(2.5, 4.5), gap=rng.uniform(5, 10), phase=rng.uniform(0, 15),
                          wear=rng.uniform(0.55, 1.0)))
    return p, present, style


def _texture(rng, h, w, scale):
    n = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)


def _road_polygon(p, x_left, x_right, z):
    ul, v = project(x_left, z, p["v0"]); ur, _ = project(x_right, z, p["v0"])
    pts = np.concatenate([np.stack([ul, v], 1), np.stack([ur, v], 1)[::-1]])
    return np.round(pts * 8).astype(np.int32)        # 3 fractional bits for cv2 sub-pixel drawing


def render(seed):
    rng = np.random.default_rng(seed)
    p, present, style = _sample_params(rng)
    v0 = p["v0"]
    img = np.zeros((H, W, 3), np.float32)

    # sky and background
    hz = int(np.clip(v0, 0, H))
    sky_top, sky_bot = rng.uniform(140, 235, 3), rng.uniform(170, 250, 3)
    a = np.linspace(0, 1, max(hz, 1))[:, None, None]
    img[:hz] = sky_top * (1 - a) + sky_bot * a
    ground = rng.choice([np.array([60, 110, 70]), np.array([90, 120, 130]), np.array([70, 90, 95])]) * rng.uniform(0.7, 1.2)
    img[hz:] = ground + 18 * _texture(rng, H - hz, W, 6)[..., None]
    for _ in range(rng.integers(0, 6)):                                  # trees / buildings on the horizon
        x0, wb, hb = rng.integers(0, W), rng.integers(20, 120), rng.integers(5, 50)
        cv2.rectangle(img, (int(x0), int(v0 - hb)), (int(x0 + wb), int(v0 + 2)), rng.uniform(30, 160, 3).tolist(), -1)

    # road surface
    z = np.geomspace(3.0, p["zmax"], 200)
    outer_l = [s for s in range(N_SLOTS) if present[s]][0]
    outer_r = [s for s in range(N_SLOTS) if present[s]][-1]
    shoulder = rng.uniform(0.3, 2.0, 2)
    xl = lane_x(p, outer_l, z) - shoulder[0]; xr = lane_x(p, outer_r, z) + shoulder[1]
    road = np.zeros((H, W), np.uint8)
    cv2.fillPoly(road, [_road_polygon(p, xl, xr, z)], 255, cv2.LINE_AA, shift=3)
    road = road.astype(np.float32)[..., None] / 255
    base = rng.uniform(60, 135)
    asphalt = base + 10 * _texture(rng, H, W, 2) + 6 * _texture(rng, H, W, 16)
    tint = rng.uniform(-6, 6, 3)
    img = img * (1 - road) + (asphalt[..., None] + tint) * road

    # repair patches and cracks (distractors)
    for _ in range(rng.integers(0, 4)):
        zc, xc = rng.uniform(6, 40), rng.uniform(-6, 6)
        zz = np.array([zc, zc, zc + rng.uniform(2, 8), zc + rng.uniform(2, 8)])
        xx = np.array([xc, xc + rng.uniform(1, 3), xc + rng.uniform(1, 3), xc])
        u, v = project(xx, zz, v0)
        cv2.fillPoly(img, [np.round(np.stack([u, v], 1) * 8).astype(np.int32)],
                     [base + rng.uniform(-35, 35)] * 3, cv2.LINE_AA, shift=3)
    for _ in range(rng.integers(0, 4)):
        zz = np.geomspace(rng.uniform(4, 15), rng.uniform(20, 60), 30)
        xx = rng.uniform(-5, 5) + np.cumsum(rng.normal(0, 0.08, 30))
        u, v = project(xx, zz, v0)
        cv2.polylines(img, [np.round(np.stack([u, v], 1) * 8).astype(np.int32)], False,
                      [base * rng.uniform(0.3, 0.7)] * 3, int(rng.integers(1, 3)), cv2.LINE_AA, shift=3)

    # lane markings
    for s in range(N_SLOTS):
        if not present[s]:
            continue
        st = style[s]
        col = (YELLOW if st["yellow"] else WHITE) * rng.uniform(0.8, 1.05)
        col = base + (col - base) * st["wear"]
        zs = np.geomspace(3.0, p["zmax"], 400)
        if st["dashed"]:
            on = ((zs + st["phase"]) % (st["dash"] + st["gap"])) < st["dash"]
        else:
            on = np.ones_like(zs, bool)
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for a0, a1 in zip(edges[::2], edges[1::2]):
            seg = zs[a0:a1]
            if len(seg) < 2:
                continue
            xc = lane_x(p, s, seg)
            cv2.fillPoly(img, [_road_polygon(p, xc - st["width"] / 2, xc + st["width"] / 2, seg)],
                         col.tolist(), cv2.LINE_AA, shift=3)

    # vehicles (occlude the markings)
    for _ in range(rng.integers(0, 4)):
        zc = rng.uniform(9, 45)
        lane = rng.integers(0, 3) - 1
        xc = p["w"] * lane - p["d"] + p["t"] * zc + 0.5 * p["c"] * zc ** 2 + rng.uniform(-0.6, 0.6)
        wv, hv = rng.uniform(1.7, 2.1), rng.uniform(1.3, 2.0)
        ul, vb = project(xc - wv / 2, zc, v0); ur, _ = project(xc + wv / 2, zc, v0)
        vt = vb - F * hv / zc
        body = rng.uniform(20, 200, 3)
        cv2.rectangle(img, (int(ul), int(vt)), (int(ur), int(vb)), body.tolist(), -1)
        cv2.rectangle(img, (int(ul), int(vt + 0.15 * (vb - vt))), (int(ur), int(vt + 0.45 * (vb - vt))),
                      (body * 0.35).tolist(), -1)
        cv2.rectangle(img, (int(ul), int(vb - 2)), (int(ur), int(vb + 1)), (15, 15, 15), -1)

    img += rng.normal(0, 2.0, img.shape)
    image = np.clip(img, 0, 255).astype(np.uint8)

    # ground truth at H_SAMPLES rows
    lanes = np.full((N_SLOTS, len(H_SAMPLES)), -2.0)
    Zr = row_to_z(H_SAMPLES, v0)
    for s in range(N_SLOTS):
        if not present[s]:
            continue
        ok = (Zr >= 3.0) & (Zr <= p["zmax"])
        u, _ = project(lane_x(p, s, Zr[ok]), Zr[ok], v0)
        col = np.full(len(H_SAMPLES), -2.0)
        col[ok] = u
        col[(col < 0) | (col >= W)] = -2
        lanes[s] = col
    present = present & ((lanes >= 0).sum(1) >= 2)
    lanes[~present] = -2
    return Scene(image, lanes, present, v0, dict(p, style=style))


def lane_mask(lanes, scale=0.5, thickness=2):
    """Per-pixel slot labels (0 = background) at `scale` of the full resolution, drawn from the ground truth."""
    h, w = int(H * scale), int(W * scale)
    m = np.zeros((h, w), np.uint8)
    for s in range(lanes.shape[0]):
        ok = lanes[s] >= 0
        if ok.sum() < 2:
            continue
        pts = np.stack([lanes[s][ok] * scale, H_SAMPLES[ok] * scale], 1)
        # extend to the bottom row along the last segment so the mask reaches the image edge
        cv2.polylines(m, [np.round(pts * 8).astype(np.int32)], False, s + 1, thickness, cv2.LINE_8, shift=3)
    return m
