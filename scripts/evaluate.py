"""Robustness benchmark: every method on the same held-out scenes under each corruption, plus CPU speed.
python scripts/evaluate.py --n 1000"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanes import classical, detector
from lanes.corruptions import BENCHMARK
from lanes.data import SPLITS
from lanes.metrics import frame_scores, summarize
from lanes.synth import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default="results/lanenet.pt")
    a = ap.parse_args()
    net = detector.load(a.model)
    seeds = list(SPLITS["test"])[: a.n]
    scenes = [render(s) for s in seeds]
    methods = ["classical_indep", "classical_joint", "dl_indep", "dl_joint"]
    res, timing = {}, {m: 0.0 for m in methods}
    for ci, (cond, fn) in enumerate(BENCHMARK.items()):
        imgs = [fn(sc.image, np.random.default_rng(s + 1_000_003 * (ci + 1))) for s, sc in zip(seeds, scenes)]
        rows = {m: [] for m in methods}
        for joint in (False, True):
            m = "classical_" + ("joint" if joint else "indep")
            t = time.perf_counter()
            for im, sc in zip(imgs, scenes):
                rows[m].append(frame_scores(classical.detect(im, joint=joint)[0], sc.lanes))
            timing[m] += time.perf_counter() - t
            m = "dl_" + ("joint" if joint else "indep")
            t = time.perf_counter()
            for i in range(0, len(imgs), 50):
                preds = detector.detect(net, imgs[i:i + 50], joint=joint)
                rows[m] += [frame_scores(p, sc.lanes) for p, sc in zip(preds, scenes[i:i + 50])]
            timing[m] += time.perf_counter() - t
        res[cond] = {m: summarize(rows[m]) for m in methods}
        print(cond, {m: round(res[cond][m]["accuracy"], 4) for m in methods}, flush=True)
    n_frames = len(seeds) * len(BENCHMARK)
    fps = {m: n_frames / timing[m] for m in methods}
    os.makedirs("results", exist_ok=True)
    json.dump(dict(n_scenes=len(seeds), results=res, fps_cpu=fps), open("results/benchmark.json", "w"), indent=1)

    names = dict(classical_indep="Classical CV, independent fits", classical_joint="Classical CV + parallel-lane model",
                 dl_indep="LaneNet, independent fits", dl_joint="LaneNet + parallel-lane model")
    conds = list(BENCHMARK)
    md = [f"Held-out scenes: {len(seeds)} per condition. TuSimple accuracy (higher is better).", "",
          "| Method | " + " | ".join(conds) + " | Mean F1 | FPS (CPU) |", "|---" * (len(conds) + 3) + "|"]
    for m in methods:
        f1 = np.mean([res[c][m]["f1"] for c in conds])
        md.append(f"| {names[m]} | " + " | ".join(f"{100 * res[c][m]['accuracy']:.1f}%" for c in conds)
                  + f" | {f1:.3f} | {fps[m]:.0f} |")
    open("results/benchmark.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
