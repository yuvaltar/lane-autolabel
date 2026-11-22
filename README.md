# Lane Auto-Labeling: Deep Learning + Classical CV + Geometry

Detects lane boundaries in road images and turns them into ready-to-use labels, comparing a small PyTorch
segmentation network with a classical computer-vision pipeline. Both share a geometric lane model and an evaluation
tool that measures accuracy, robustness and speed.

The data is synthetic, generated with exact ground truth (no public driving dataset was reachable from the build
environment). The generator randomizes road geometry, lane count, marking type and wear, occluding vehicles,
distractors (cracks, repair patches) and the camera pitch.

## Pipeline

1. **Data generation** (`lanes/synth.py`): a calibrated pinhole camera over a flat road. Every scene is defined by its
   seed, so the train, validation and test splits are disjoint seed ranges. Labels use the TuSimple format.
2. **Deep learning** (`lanes/model.py`, `scripts/train.py`): a compact U-Net (0.56M parameters) that classifies each
   pixel as background or one of 4 lane slots (ego-left, ego-right and one more lane on each side). Training uses
   corruption augmentation and horizontal flips that swap the left and right slots.
3. **Classical CV** (`lanes/classical.py`): top-hat ridge detection, inverse perspective mapping to the road plane, a
   lateral histogram to find where each lane starts, then sliding windows that follow each lane forward.
4. **Geometric model** (`lanes/geometry.py`): RANSAC per lane, then a joint fit in which all lanes share heading and
   curvature (lanes are parallel) and differ only in lateral offset. Residuals are measured in image pixels.
5. **Evaluation** (`scripts/evaluate.py`): TuSimple accuracy, false-positive rate, false-negative rate and F1, on
   held-out scenes under 6 conditions: clean, shadow, noise, motion blur, low light and glare. Also reports CPU speed.
6. **Auto-labeling** (`scripts/autolabel.py`): pre-labels a batch of frames. Each frame gets a confidence score
   (network probability × RANSAC inlier ratio × agreement between the joint and per-lane fits). The acceptance
   threshold is calibrated on the validation split and then applied unchanged to the test split. Accepted labels are
   exported as TuSimple JSON lines, and low-confidence frames go to a review queue with their pre-labels.

## Results

Held-out scenes: 1000 per condition. TuSimple accuracy (higher is better).

| Method | clean | shadow | noise | blur | low_light | glare | Mean F1 | FPS (CPU) |
|---|---|---|---|---|---|---|---|---|
| Classical CV, independent fits | 87.3% | 78.3% | 70.4% | 61.2% | 85.6% | 86.8% | 0.675 | 61 |
| Classical CV + parallel-lane model | 90.8% | 84.5% | 78.3% | 67.2% | 89.2% | 90.2% | 0.774 | 61 |
| LaneNet, independent fits | 98.0% | 97.9% | 98.0% | 97.5% | 97.8% | 98.0% | 0.996 | 43 |
| LaneNet + parallel-lane model | 100.0% | 99.9% | 99.9% | 99.8% | 99.9% | 99.9% | 0.998 | 43 |

FPS measured on a 2-core cloud VM while another job was running, so treat it as a rough figure.

**Auto-labeling** (1,000 test frames, mixed conditions; threshold calibrated on validation for at most 2% error):
99.4% of frames were labeled with every lane correct. 99.5% of frames were auto-accepted with 0.4% error, and 0.5%
were sent to review.

Limitations: on this synthetic data the network is already almost always right, so the confidence filter has little
to remove. Real recordings (night, rain, worn paint, merges) would need real data and a larger model.

## Run

```
pip install -r requirements.txt
python scripts/train.py --epochs 8 --n-train 6000     # about 25 min on 2 CPU cores
python scripts/evaluate.py --n 1000
python scripts/autolabel.py --max-error 0.02
pytest -q
```
