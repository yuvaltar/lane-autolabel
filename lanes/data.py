"""Dataset splits as disjoint seed ranges, and dataset generation."""
import cv2
import numpy as np

from .corruptions import augment
from .model import IN_H, IN_W
from .synth import lane_mask, render

SPLITS = {"train": range(0, 8000), "val": range(100_000, 100_500), "test": range(200_000, 201_000)}


def small(img):
    return cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)


def make_train(seeds, aug=True):
    X = np.zeros((len(seeds), IN_H, IN_W, 3), np.uint8)
    Y = np.zeros((len(seeds), IN_H, IN_W), np.uint8)
    for i, s in enumerate(seeds):
        sc = render(s)
        img = augment(sc.image, np.random.default_rng(s + 7_000_000)) if aug else sc.image
        X[i] = small(img)
        Y[i] = lane_mask(sc.lanes, scale=IN_H / 256)
    return X, Y
