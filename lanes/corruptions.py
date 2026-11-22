"""Image corruptions for robustness evaluation (fixed severity) and training augmentation (random severity)."""
import cv2
import numpy as np

from .camera import H, W


def shadow(img, rng, strength=None):
    """Tree/building shadows: 1-3 random polygons darkened on the road."""
    out = img.astype(np.float32)
    m = np.zeros((H, W), np.float32)
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0, W), rng.uniform(H * 0.5, H)
        ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
        r = rng.uniform(30, 140, 7)
        pts = np.stack([cx + r * np.cos(ang), cy + 0.6 * r * np.sin(ang)], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    k = strength if strength is not None else rng.uniform(0.35, 0.6)
    out *= (1 - (1 - k) * m)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def noise(img, rng, sigma=None):
    s = sigma if sigma is not None else rng.uniform(5, 22)
    return np.clip(img + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def blur(img, rng, size=None):
    """Motion blur along a random direction."""
    n = int(size if size is not None else rng.integers(5, 12))
    k = np.zeros((n, n), np.float32)
    k[n // 2] = 1
    rot = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), rng.uniform(0, 180), 1)
    k = cv2.warpAffine(k, rot, (n, n)); k /= k.sum()
    return cv2.filter2D(img, -1, k)


def low_light(img, rng, gain=None):
    g = gain if gain is not None else rng.uniform(0.25, 0.6)
    out = 255 * (img / 255.0) ** 1.8 * g
    return np.clip(out + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def glare(img, rng, amount=None):
    """Low sun: bright blob near the horizon and washed-out contrast."""
    a = amount if amount is not None else rng.uniform(0.4, 0.8)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = rng.uniform(0.2 * W, 0.8 * W), rng.uniform(0.3 * H, 0.5 * H)
    blob = np.exp(-(((xx - cx) / (0.35 * W)) ** 2 + ((yy - cy) / (0.35 * H)) ** 2))
    out = img * (1 - 0.5 * a) + 255 * (0.5 * a) * blob[..., None] + 60 * a
    return np.clip(out, 0, 255).astype(np.uint8)


# Fixed-severity versions used by the robustness benchmark.
BENCHMARK = {
    "clean": lambda im, rng: im,
    "shadow": lambda im, rng: shadow(im, rng, 0.4),
    "noise": lambda im, rng: noise(im, rng, 20),
    "blur": lambda im, rng: blur(im, rng, 9),
    "low_light": lambda im, rng: low_light(im, rng, 0.35),
    "glare": lambda im, rng: glare(im, rng, 0.7),
}

_AUG = [shadow, noise, blur, low_light, glare]


def augment(img, rng):
    """Training augmentation: half of the images get one or two random corruptions at random severity."""
    if rng.random() < 0.5:
        for i in rng.choice(len(_AUG), rng.integers(1, 3), replace=False):
            img = _AUG[i](img, rng)
    return img
