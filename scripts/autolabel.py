"""Pre-label a batch of frames and split it into auto-accepted labels and a review queue.
python scripts/autolabel.py --max-error 0.02

Frames come from a mix of all benchmark conditions (clean, shadow, noise, blur, low light, glare), like a real
recording batch. Writes results/autolabel.json (metrics), results/labels_accepted.jsonl (TuSimple format) and
results/review_queue.jsonl (pre-labels for annotators)."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanes import detector
from lanes.autolabel import apply, calibrate, risk_coverage, to_tusimple
from lanes.corruptions import BENCHMARK
from lanes.data import SPLITS
from lanes.metrics import slot_correct
from lanes.synth import render

CONDS = list(BENCHMARK)


def run(net, seeds):
    conf, correct, labels = [], [], []
    for i in range(0, len(seeds), 50):
        batch = seeds[i:i + 50]
        scenes = [render(s) for s in batch]
        conds = [CONDS[s % len(CONDS)] for s in batch]
        imgs = [BENCHMARK[c](sc.image, np.random.default_rng(s + 99)) for s, sc, c in zip(batch, scenes, conds)]
        for s, sc, c, (lanes, cf) in zip(batch, scenes, conds, detector.detect(net, imgs, with_confidence=True)):
            conf.append(cf); correct.append(slot_correct(lanes, sc.lanes)); labels.append((f"{s}_{c}.png", lanes, cf))
    return np.array(conf), np.array(correct), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-error", type=float, default=0.02)
    ap.add_argument("--model", default="results/lanenet.pt")
    a = ap.parse_args()
    net = detector.load(a.model)
    cv, okv, _ = run(net, list(SPLITS["val"]))
    thr, cov_v, err_v = calibrate(cv, okv, a.max_error)
    ct, okt, labels = run(net, list(SPLITS["test"]))
    test = apply(ct, okt, thr)
    out = dict(target_error=a.max_error, threshold=thr, val=dict(coverage=cov_v, error_accepted=err_v),
               test=test, test_frames=len(ct), risk_coverage_test=risk_coverage(ct, okt))
    json.dump(out, open("results/autolabel.json", "w"), indent=1)
    with open("results/labels_accepted.jsonl", "w") as fa, open("results/review_queue.jsonl", "w") as fr:
        for name, lanes, cf in labels:
            (fa if cf >= thr else fr).write(to_tusimple(name, lanes) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "risk_coverage_test"}, indent=1))


if __name__ == "__main__":
    main()
