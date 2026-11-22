import numpy as np

from lanes import classical
from lanes.autolabel import apply, calibrate
from lanes.camera import H_SAMPLES
from lanes.corruptions import BENCHMARK, augment
from lanes.data import SPLITS
from lanes.metrics import frame_scores, slot_correct, summarize
from lanes.synth import lane_mask, render


def test_splits_are_disjoint():
    tr, va, te = (set(SPLITS[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_scenes_are_deterministic():
    a, b = render(123), render(123)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.lanes, b.lanes)
    assert not np.array_equal(render(124).image, a.image)


def test_ground_truth_sits_on_painted_markings():
    """Near rows of solid lanes: the label points must land on detected marking pixels (cars may hide a few)."""
    hits = total = 0
    for seed in range(40):
        sc = render(seed)
        u, v = classical.marking_pixels(sc.image)
        on = set(zip((u - 0.5).astype(int), (v - 0.5).astype(int)))
        for s in np.flatnonzero(sc.present):
            if sc.params["style"][s]["dashed"]:
                continue
            for x, y in zip(sc.lanes[s], H_SAMPLES):
                if x >= 0 and y > 190:
                    total += 1
                    hits += any((int(x) + dx, int(y) + dy) in on for dx in (-2, -1, 0, 1, 2) for dy in (-1, 0))
    assert total > 300 and hits / total > 0.9


def test_mask_has_one_class_per_present_lane():
    sc = render(7)
    m = lane_mask(sc.lanes)
    assert set(np.unique(m)) - {0} == {s + 1 for s in np.flatnonzero(sc.present)}


def test_corruptions_keep_shape_and_type():
    img = render(1).image
    for name, fn in BENCHMARK.items():
        out = fn(img, np.random.default_rng(0))
        assert out.shape == img.shape and out.dtype == np.uint8
    assert augment(img, np.random.default_rng(5)).shape == img.shape


def test_metrics_perfect_shifted_missing_extra():
    gt = render(9).lanes
    r = frame_scores(gt, gt)
    assert r["acc"] == r["n_gt"] and r["fp"] == 0 and r["fn"] == 0 and slot_correct(gt, gt)
    shifted = np.where(gt >= 0, gt + 15, -2)
    assert frame_scores(shifted, gt)["matched"] == 0
    missing = gt.copy(); missing[1] = -2
    assert frame_scores(missing, gt)["fn"] == 1 and not slot_correct(missing, gt)
    assert summarize([frame_scores(gt, gt)])["f1"] == 1.0


def test_classical_baseline_regression():
    rows = [frame_scores(classical.detect(render(s).image)[0], render(s).lanes) for s in range(200_000, 200_030)]
    assert summarize(rows)["accuracy"] > 0.85


def test_calibration_respects_target_error():
    rng = np.random.default_rng(0)
    conf = rng.random(2000)
    correct = rng.random(2000) < conf            # higher confidence -> more often correct
    thr, cov, err = calibrate(conf, correct, max_error=0.05)
    assert err <= 0.05 and 0 < cov < 1
    res = apply(conf, correct, thr)
    assert abs(res["coverage"] - cov) < 1e-9 and res["error_accepted"] <= 0.05
