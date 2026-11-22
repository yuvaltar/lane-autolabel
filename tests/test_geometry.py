import numpy as np
import pytest

from lanes.camera import H_SAMPLES, project, row_to_z, unproject
from lanes.geometry import fit_joint, lanes_from_points, ransac_lane, sample_lane
from lanes.synth import lane_x, render


def test_project_unproject_roundtrip():
    X = np.array([-5.0, -1.2, 0.0, 2.5]); Z = np.array([4.0, 10.0, 30.0, 70.0])
    u, v = project(X, Z)
    X2, Z2 = unproject(u, v)
    assert np.allclose(X, X2) and np.allclose(Z, Z2)


def test_rows_above_horizon_are_infinitely_far():
    assert np.isinf(row_to_z(np.array([10.0]))).all()


def test_joint_fit_recovers_shared_heading_and_curvature():
    rng = np.random.default_rng(1)
    t, c, offsets = 0.02, 1e-3, {0: -5.4, 1: -1.8, 2: 1.8}
    groups = {}
    for s, o in offsets.items():
        Z = rng.uniform(5, 60, 80)
        groups[s] = (o + t * Z + 0.5 * c * Z ** 2, Z)
    coefs = fit_joint(groups)
    for s, o in offsets.items():
        assert np.allclose(coefs[s], [o, t, c], atol=1e-9)


def test_ransac_ignores_outliers():
    rng = np.random.default_rng(2)
    Z = rng.uniform(5, 50, 200)
    X = -1.7 + 0.01 * Z + 0.5 * 8e-4 * Z ** 2
    bad = rng.random(200) < 0.3
    X[bad] += rng.choice([-1, 1], bad.sum()) * rng.uniform(0.5, 3, bad.sum())
    coef, inl = ransac_lane(X, Z, rng)
    assert np.allclose(coef, [-1.7, 0.01, 8e-4], atol=1e-6)
    assert inl[~bad].all()


def test_geometry_reproduces_the_generator_ground_truth():
    """Sampling the true road model through the geometry module gives the generator's labels (shared conventions)."""
    sc = render(5)
    p = sc.params
    for s in np.flatnonzero(sc.present):
        coef = np.array([p["w"] * [-1.5, -0.5, 0.5, 1.5][s] - p["d"], p["t"], p["c"]])
        Z = row_to_z(H_SAMPLES, sc.v0)
        assert np.allclose(lane_x(p, s, Z[np.isfinite(Z)]), (np.c_[np.ones(np.isfinite(Z).sum()), Z[np.isfinite(Z)],
                           0.5 * Z[np.isfinite(Z)] ** 2] @ coef))
        pred = sample_lane(coef, 0, sc.v0)
        ok = sc.lanes[s] >= 0
        assert np.allclose(pred[ok], sc.lanes[s][ok], atol=1e-6)


def test_lane_with_missing_far_dashes_is_extended_by_the_joint_model():
    rng = np.random.default_rng(3)
    t, c = 0.01, 5e-4
    pts = {}
    for s, (o, zmax) in {1: (-1.8, 60), 2: (1.8, 15)}.items():      # slot 2 only seen up to 15 m
        Z = np.geomspace(4, zmax, 120)
        u, v = project(o + t * Z + 0.5 * c * Z ** 2, Z)
        pts[s] = (u, v)
    joint, _ = lanes_from_points(pts, rng, joint=True)
    indep, _ = lanes_from_points(pts, rng, joint=False)
    assert (joint[2] >= 0).sum() > (indep[2] >= 0).sum()
