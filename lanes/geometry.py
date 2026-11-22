"""Lane geometry on the road plane: robust per-lane fitting and a joint parallel-lane model.

All lanes on a road share heading and curvature and differ only in lateral offset:
    X_k(Z) = o_k + t * Z + 0.5 * c * Z^2
Fitting that joint model lets well-seen lanes fix the shape of poorly-seen ones (shadows, occlusion, worn paint).
Residuals are measured in image pixels, so far-away points are not over-weighted.
"""
import numpy as np

from .camera import F, H, H_SAMPLES, V0, W, project, row_to_z, unproject


def _design(Z):
    return np.stack([np.ones_like(Z), Z, 0.5 * Z ** 2], 1)


def fit_lane(X, Z, w=None):
    """Weighted least squares for one lane: returns (o, t, c). Rows are scaled by F/Z to work in pixel units."""
    s = F / Z if w is None else w * F / Z
    A = _design(Z) * s[:, None]
    return np.linalg.lstsq(A, X * s, rcond=None)[0]


def pixel_residual(coef, X, Z):
    return np.abs(F * (X - _design(Z) @ coef) / Z)


def ransac_lane(X, Z, rng, iters=60, thresh_px=3.0, min_pts=12):
    """RANSAC over 3-point samples; returns (coef, inlier mask) or (None, None) if no consistent lane."""
    n = len(X)
    if n < min_pts:
        return None, None
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if np.ptp(Z[idx]) < 1.0:
            continue
        try:
            coef = np.linalg.solve(_design(Z[idx]), X[idx])
        except np.linalg.LinAlgError:
            continue
        inl = pixel_residual(coef, X, Z) < thresh_px
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_pts:
        return None, None
    coef = fit_lane(X[best], Z[best])
    inl = pixel_residual(coef, X, Z) < thresh_px
    return fit_lane(X[inl], Z[inl]), inl


def fit_joint(groups):
    """groups: dict slot -> (X, Z) inlier points. Returns dict slot -> (o, t, c) with shared t, c."""
    slots = sorted(groups)
    rows, rhs = [], []
    for j, s in enumerate(slots):
        X, Z = groups[s]
        sc = F / Z
        A = np.zeros((len(X), len(slots) + 2))
        A[:, j] = 1; A[:, -2] = Z; A[:, -1] = 0.5 * Z ** 2
        rows.append(A * sc[:, None]); rhs.append(X * sc)
    sol = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    return {s: np.array([sol[j], sol[-2], sol[-1]]) for j, s in enumerate(slots)}


def sample_lane(coef, v_top, v0=V0):
    """Column of the lane at every H_SAMPLES row from v_top down to the bottom of the image (-2 elsewhere)."""
    out = np.full(len(H_SAMPLES), -2.0)
    ok = H_SAMPLES >= v_top
    Z = row_to_z(H_SAMPLES[ok], v0)
    u, _ = project(_design(Z) @ coef, Z, v0)
    u[(u < 0) | (u >= W)] = -2
    out[ok] = u
    return out


def lanes_from_points(points, rng, joint=True, n_slots=4, min_pts=12):
    """points: dict slot -> (u, v) pixel arrays (full resolution). Returns (lanes N_SLOTS x len(H_SAMPLES), info)."""
    lanes = np.full((n_slots, len(H_SAMPLES)), -2.0)
    groups, tops, info = {}, {}, {}
    for s, (u, v) in points.items():
        keep = v > V0 + 4
        X, Z = unproject(u[keep], v[keep])
        coef, inl = ransac_lane(X, Z, rng, min_pts=min_pts)
        if coef is None:
            continue
        groups[s] = (X[inl], Z[inl])
        tops[s] = v[keep][inl].min()
        info[s] = dict(inlier_ratio=float(inl.mean()), n=int(inl.sum()),
                       rms_px=float(np.sqrt(np.mean(pixel_residual(coef, X[inl], Z[inl]) ** 2))))
    if not groups:
        return lanes, info
    if joint and len(groups) >= 2:
        # parallel lanes are visible over the same stretch of road: a lane whose far dashes were missed is
        # extended to the farthest row any lane reached
        coefs = fit_joint(groups)
        tops = {s: min(tops.values()) for s in tops}
    else:
        coefs = {s: fit_lane(*groups[s]) for s in groups}
    for s, c in coefs.items():
        lanes[s] = sample_lane(c, tops[s])
    return lanes, info
