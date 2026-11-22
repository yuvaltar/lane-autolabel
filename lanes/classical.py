"""Classical computer-vision baseline: ridge detection -> road-plane projection -> sliding windows -> geometric fit.

1. Markings are bright ridges narrower than ~20 px: a horizontal white top-hat isolates them from asphalt, patches and
   shadows' broad brightness changes; a local adaptive threshold keeps the strongest ridges.
2. Candidate pixels are projected onto the road plane with the calibrated camera (inverse perspective mapping).
3. Lane bases come from a lateral histogram near the car; sliding windows follow each lane forward.
4. Bases are assigned to slots by side and order (nearest left = ego-left, ...), then fitted with the same RANSAC +
   joint parallel-lane model as the deep-learning pipeline.
"""
import cv2
import numpy as np

from .camera import H, V0, unproject
from .geometry import lanes_from_points

TOPHAT = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))


def marking_pixels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, TOPHAT).astype(np.float32)
    th[: int(V0) + 6] = 0
    # adaptive: relative to the local contrast in a band of rows, so low light and glare still work
    thr = np.maximum(12.0, 0.45 * np.percentile(th[int(V0) + 6:], 99.5))
    v, u = np.nonzero(th > thr)
    return u.astype(float) + 0.5, v.astype(float) + 0.5


def _peaks(hist, centers, min_count, min_sep):
    order = np.argsort(hist)[::-1]
    out = []
    for i in order:
        if hist[i] < min_count:
            break
        if all(abs(centers[i] - centers[j]) >= min_sep for j in out):
            out.append(i)
    return [centers[i] for i in out]


def _follow(X, Z, x0, zbins, half=0.45):
    """Sliding windows along Z starting at lateral position x0; returns indices of collected points."""
    taken, cx, hist = [], x0, []
    misses = 0
    for z0, z1 in zip(zbins[:-1], zbins[1:]):
        if len(hist) >= 2:                                 # extrapolate the centre from the windows found so far
            zz, xx = np.array(hist).T
            k = np.polyfit(zz, xx, 1 if len(hist) < 4 else 2)
            cx = np.polyval(k, 0.5 * (z0 + z1))
        win = half * (1 + 0.02 * z0)
        sel = np.flatnonzero((Z >= z0) & (Z < z1) & (np.abs(X - cx) < win))
        if len(sel) >= 3:
            taken.append(sel); hist.append((np.median(Z[sel]), np.median(X[sel]))); misses = 0
        elif hist:                                         # lanes can start beyond the bottom of the image
            misses += 1
            if misses >= 7:
                break
    return np.concatenate(taken) if taken else np.array([], int)


def detect(img, rng=None, joint=True):
    rng = rng if rng is not None else np.random.default_rng(0)
    u, v = marking_pixels(img)
    keep = v > V0 + 6
    u, v = u[keep], v[keep]
    X, Z = unproject(u, v)
    near = (Z < 18) & (np.abs(X) < 8)
    centers = np.arange(-8, 8, 0.1) + 0.05
    # weight by distance: a far marking covers fewer pixels than a near one of the same length
    hist, _ = np.histogram(X[near], bins=np.arange(-8, 8.01, 0.1), weights=Z[near] / 4)
    hist = np.convolve(hist, np.ones(5), "same")
    bases = sorted(_peaks(hist, centers, min_count=40, min_sep=2.4))
    left = sorted([b for b in bases if b < 0], reverse=True)[:2]
    right = sorted([b for b in bases if b >= 0])[:2]
    zbins = np.geomspace(3.2, 70, 22)
    points = {}
    for slot, b in [(1, left[0] if left else None), (0, left[1] if len(left) > 1 else None),
                    (2, right[0] if right else None), (3, right[1] if len(right) > 1 else None)]:
        if b is None:
            continue
        idx = _follow(X, Z, b, zbins)
        if len(idx):
            rows, inv = np.unique(v[idx], return_inverse=True)        # one centre per image row
            points[slot] = (np.bincount(inv, u[idx]) / np.bincount(inv), rows)
    lanes, info = lanes_from_points(points, rng, joint=joint, min_pts=6)
    return lanes, info
