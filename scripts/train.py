"""Train LaneNet on synthetic scenes.  python scripts/train.py --epochs 12"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanes.data import SPLITS, make_train
from lanes.model import LaneNet, to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-train", type=int, default=len(SPLITS["train"]))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/lanenet.pt")
    a = ap.parse_args()
    torch.manual_seed(0)

    t0 = time.time()
    X, Y = make_train(list(SPLITS["train"])[: a.n_train])
    Xv, Yv = make_train(list(SPLITS["val"]), aug=False)
    print(f"data: {len(X)} train, {len(Xv)} val, generated in {time.time() - t0:.0f}s", flush=True)

    net = LaneNet()
    print("parameters:", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = a.epochs * (len(X) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    weight = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0])
    log = []
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(len(X)); tl = 0.0; t1 = time.time()
        for i in range(len(X) // a.batch):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            x = to_tensor(X[idx]); y = torch.from_numpy(Y[idx]).long()
            if rng.random() < 0.5:                       # horizontal flip: mirror image and swap slots 1<->4, 2<->3
                x = x.flip(-1); y = y.flip(-1)
                y = torch.where(y > 0, 5 - y, y)
            loss = Fn.cross_entropy(net(x), y, weight=weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item()
        net.eval(); inter = np.zeros(5); union = np.zeros(5)
        with torch.no_grad():
            for i in range(0, len(Xv), 50):
                p = net(to_tensor(Xv[i:i + 50])).argmax(1).numpy(); y = Yv[i:i + 50]
                for c in range(1, 5):
                    inter[c] += ((p == c) & (y == c)).sum(); union[c] += ((p == c) | (y == c)).sum()
        miou = float(np.mean(inter[1:] / np.maximum(union[1:], 1)))
        log.append(dict(epoch=ep + 1, loss=tl / (len(X) // a.batch), val_lane_miou=miou, sec=time.time() - t1))
        print(log[-1], flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        torch.save(net.state_dict(), a.out)
    json.dump(log, open(os.path.join(os.path.dirname(a.out), "train_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
