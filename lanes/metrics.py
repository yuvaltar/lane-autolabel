"""TuSimple-style lane metrics.

A predicted lane matches a ground-truth lane when at least 85% of the ground-truth rows are within a pixel threshold
(20 px at TuSimple's 1280-px width, scaled here to 8 px at 512 px, widened by 1/cos of the lane's angle).
"""
import numpy as np

from .camera import H_SAMPLES

PIXEL_THRESH = 8.0
PT_THRESH = 0.85


def _angle(xs):
    ok = xs >= 0
    if ok.sum() > 1:
        k = np.polyfit(H_SAMPLES[ok], xs[ok], 1)[0]
        return np.arctan(k)
    return 0.0


def lane_accuracy(pred, gt):
    ok = gt >= 0
    if ok.sum() == 0:
        return 0.0
    thr = PIXEL_THRESH / np.cos(_angle(gt))
    hit = (pred[ok] >= 0) & (np.abs(pred[ok] - gt[ok]) < thr)
    return float(hit.mean())


def frame_scores(pred_lanes, gt_lanes):
    """Returns dict: accuracy sum, matched GT count, GT count, pred count, false positives, false negatives."""
    gts = [g for g in gt_lanes if (g >= 0).sum() >= 2]
    preds = [p for p in pred_lanes if (p >= 0).sum() >= 2]
    used = set()
    acc_sum, matched = 0.0, 0
    for g in gts:
        accs = [lane_accuracy(p, g) for p in preds]
        if accs:
            j = int(np.argmax(accs)); acc_sum += accs[j]
            if accs[j] >= PT_THRESH:
                matched += 1; used.add(j)
    fp = sum(1 for j, p in enumerate(preds) if j not in used)
    return dict(acc=acc_sum, matched=matched, n_gt=len(gts), n_pred=len(preds), fp=fp, fn=len(gts) - matched)


def slot_correct(pred_lanes, gt_lanes):
    """Frame-level label quality for auto-labeling: every slot present in exactly the right place (no extra lanes)."""
    for p, g in zip(pred_lanes, gt_lanes):
        gp, pp = (g >= 0).sum() >= 2, (p >= 0).sum() >= 2
        if gp != pp:
            return False
        if gp and lane_accuracy(p, g) < PT_THRESH:
            return False
    return True


def summarize(rows):
    acc = sum(r["acc"] for r in rows) / max(1, sum(r["n_gt"] for r in rows))
    fp = sum(r["fp"] for r in rows) / max(1, sum(r["n_pred"] for r in rows))
    fn = sum(r["fn"] for r in rows) / max(1, sum(r["n_gt"] for r in rows))
    tp = sum(r["matched"] for r in rows)
    prec = tp / max(1, sum(r["n_pred"] for r in rows)); rec = tp / max(1, sum(r["n_gt"] for r in rows))
    return dict(accuracy=acc, fp_rate=fp, fn_rate=fn, f1=2 * prec * rec / max(1e-9, prec + rec))
