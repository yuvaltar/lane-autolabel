"""Pinhole camera over a flat road, and the image <-> road-plane mapping used by every pipeline.

Road coordinates: X is lateral (metres, right is positive), Z is forward distance (metres).
A road point (X, Z) projects to  u = U0 + F * X / Z,  v = v0 + F * H_CAM / Z.
"""
import numpy as np

W, H = 512, 256          # full image size (pixels)
F = 360.0                # focal length (pixels)
U0 = W / 2.0             # principal point, column
V0 = 100.0               # nominal horizon row (pitch jitter moves the true horizon by a few pixels)
H_CAM = 1.5              # camera height above the road (m)

# Rows at which lanes are labelled and evaluated (TuSimple-style "h_samples").
H_SAMPLES = np.arange(116, H, 4)


def project(X, Z, v0=V0):
    """Road point(s) -> image (u, v)."""
    X = np.asarray(X, float); Z = np.asarray(Z, float)
    return U0 + F * X / Z, v0 + F * H_CAM / Z


def row_to_z(v, v0=V0):
    """Image row -> forward distance of the road point seen on that row (inf at/above the horizon)."""
    v = np.asarray(v, float)
    with np.errstate(divide="ignore"):
        return np.where(v > v0, F * H_CAM / np.maximum(v - v0, 1e-9), np.inf)


def unproject(u, v, v0=V0):
    """Image pixel(s) below the horizon -> road point(s) (X, Z)."""
    Z = row_to_z(v, v0)
    return (np.asarray(u, float) - U0) * Z / F, Z
