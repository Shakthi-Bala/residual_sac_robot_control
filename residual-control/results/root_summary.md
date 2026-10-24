## Residual control study

**Question:** Can bounded residual SAC improve a tuned controller under hidden payload and delay?

A simulation study on a six-joint UR16e compares Cartesian PI, standalone SAC, and residual SAC with three training seeds, equal budgets, and 1,008 matched held-out episodes.

| Method | Success | RMSE after 1 s | Final error | Command acceleration RMS | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cartesian PI | 89.6% | 11.2 mm | 7.1 mm | 0.7 rad/s² | 0.0% |
| Standalone SAC | 9.0 ± 12.2% | 44.9 ± 14.1 mm | 66.9 ± 39.6 mm | 2.6 ± 0.0 rad/s² | 33.3 ± 23.6% |
| Residual SAC | 89.8 ± 0.4% | 10.6 ± 1.1 mm | 7.9 ± 1.5 mm | 0.9 ± 0.3 rad/s² | 0.0 ± 0.0% |

Residual SAC changes aggregate RMSE by +5.8% relative to PI (positive means improvement). 2/3 residual seeds retain the initial zero-correction checkpoint. Retaining PI is not learned improvement.

[Full protocol and results](residual-control/README.md) · [Report PDF](residual-control/results/report.pdf) · [Learning curves](residual-control/results/learning.png) · [Per-episode data](residual-control/results/episodes.csv)
