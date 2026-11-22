"""Auto-labeling: accept confident pre-labels automatically, send the rest to annotators.

The acceptance threshold is calibrated on the validation split for a target error rate among accepted frames, then
applied unchanged to the test split, so the reported numbers are not tuned on the data they are measured on.
"""
import json

import numpy as np

from .camera import H_SAMPLES


def calibrate(conf, correct, max_error=0.02, min_accepted=50):
    """Lowest threshold whose accepted frames have error <= max_error. Returns (threshold, coverage, error)."""
    conf = np.asarray(conf); correct = np.asarray(correct, bool)
    order = np.argsort(-conf)
    errors = np.cumsum(~correct[order]) / np.arange(1, len(conf) + 1)
    best = None
    for k in range(min_accepted, len(conf) + 1):
        if errors[k - 1] <= max_error and (k == len(conf) or conf[order[k - 1]] > conf[order[k]]):
            best = k
    if best is None:
        return float("inf"), 0.0, 0.0
    return float(conf[order[best - 1]]), best / len(conf), float(errors[best - 1])


def apply(conf, correct, threshold):
    conf = np.asarray(conf); correct = np.asarray(correct, bool)
    acc = conf >= threshold
    return dict(coverage=float(acc.mean()),
                error_accepted=float((~correct[acc]).mean()) if acc.any() else 0.0,
                error_all=float((~correct).mean()),
                review_queue=float((~acc).mean()))


def risk_coverage(conf, correct, points=20):
    conf = np.asarray(conf); correct = np.asarray(correct, bool)
    order = np.argsort(-conf)
    out = []
    for q in np.linspace(0.05, 1.0, points):
        k = max(1, int(round(q * len(conf))))
        out.append((k / len(conf), float((~correct[order[:k]]).mean())))
    return out


def to_tusimple(name, lanes):
    """One label line in TuSimple format."""
    return json.dumps(dict(raw_file=name, h_samples=H_SAMPLES.tolist(),
                           lanes=[[int(round(x)) if x >= 0 else -2 for x in lane] for lane in lanes]))
