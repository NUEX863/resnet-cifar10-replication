# Measured CIFAR-10 comparison

Final fixed-budget checkpoints; no selection by test accuracy.

| Model | Train error (%) | Test error (%) |
|---|---:|---:|
| plain20 | 0.47 | 9.38 |
| plain56 | 3.86 | 13.82 |
| resnet20 | 0.33 | 8.91 |
| resnet56 | 0.01 | 6.80 |

Differences below are depth 56 minus depth 20, in percentage points.
- plain train_error: +3.38
- plain test_error: +4.44
- resnet train_error: -0.32
- resnet test_error: -2.11

Positive Plain training difference supports degradation in this run. Negative ResNet differences support improvement with depth. A single seed does not establish robustness; report contradictory outcomes as measured.
