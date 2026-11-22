Held-out scenes: 1000 per condition. TuSimple accuracy (higher is better).

| Method | clean | shadow | noise | blur | low_light | glare | Mean F1 | FPS (CPU) |
|---|---|---|---|---|---|---|---|---|
| Classical CV, independent fits | 87.3% | 78.3% | 70.4% | 61.2% | 85.6% | 86.8% | 0.675 | 61 |
| Classical CV + parallel-lane model | 90.8% | 84.5% | 78.3% | 67.2% | 89.2% | 90.2% | 0.774 | 61 |
| LaneNet, independent fits | 98.0% | 97.9% | 98.0% | 97.5% | 97.8% | 98.0% | 0.996 | 43 |
| LaneNet + parallel-lane model | 100.0% | 99.9% | 99.9% | 99.8% | 99.9% | 99.9% | 0.998 | 43 |
