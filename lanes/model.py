"""Compact U-Net: image (3 x 128 x 256) -> per-pixel scores for background + 4 lane slots."""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

N_CLASSES = 5
IN_H, IN_W = 128, 256


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LaneNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 96)):
        super().__init__()
        # full-resolution stem keeps thin far-away markings; the U-Net itself runs at half resolution
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True))
        self.down = nn.Sequential(nn.Conv2d(8, ch[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch[0]),
                                  nn.ReLU(inplace=True))
        self.enc = nn.ModuleList()
        cin = ch[0]
        for c in ch:
            self.enc.append(_block(cin, c)); cin = c
        # wide horizontal context at the bottleneck: lane slots depend on where the other lanes are
        self.ctx = nn.Sequential(nn.Conv2d(ch[-1], ch[-1], (1, 9), padding=(0, 4)), nn.ReLU(inplace=True),
                                 nn.Conv2d(ch[-1], ch[-1], (9, 1), padding=(4, 0)), nn.ReLU(inplace=True))
        self.dec = nn.ModuleList()
        for c_hi, c_lo in zip(ch[::-1][:-1], ch[::-1][1:]):
            self.dec.append(_block(c_hi + c_lo, c_lo))
        self.up = nn.Sequential(nn.Conv2d(ch[0] + 8, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(16, N_CLASSES, 1)

    def forward(self, x):
        s0 = self.stem(x)
        x = self.down(s0)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x); x = Fn.max_pool2d(x, 2)
        x = x + self.ctx(x)
        for blk, s in zip(self.dec, skips[::-1]):
            x = Fn.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, s], 1))
        x = Fn.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up(torch.cat([x, s0], 1)))


def to_tensor(images_bgr_small):
    """uint8 N x h x w x 3 (BGR) -> float tensor N x 3 x h x w in [0, 1]."""
    t = torch.from_numpy(images_bgr_small).permute(0, 3, 1, 2).float() / 255.0
    return t
