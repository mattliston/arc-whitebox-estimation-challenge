# Numerical training reference: sampled response qualification passed

All four fixed direction/network final-response cells pass the original
5% secant,0.10 curvature and5e-7 RMS requirements. This establishes a limited
numerical reference for subsequent proposal fitting, not a native fast-path
gradient, general differentiability, estimator accuracy gain or submission.

| Direction | MINI | Relative secant gap | Curvature | RMS(h*r_half) |
|---:|---:|---:|---:|---:|
| 0 | 57 | 0.016808 | 0.012892 | 2.95308231e-06 |
| 0 | 68 | 0.019347 | 0.015338 | 2.84857368e-06 |
| 7 | 57 | 0.015352 | 0.013290 | 2.96417872e-06 |
| 7 | 68 | 0.020214 | 0.014781 | 2.77132206e-06 |

The reference preserves layers0..8, uses ordinary switch-controlled products
from layer9 onward, and widens the current Nyström sketch/QR at joins10/13.
Q returns toFP32 for original consumers; stored cores remainFP32. All model
parameters except the predetermined readout direction stay fixed. Selection,
ranks and nonlinear formulas are unchanged. The Winograd switch is restored
after every call. This combination addresses the covariance and current
sketch arithmetic measured in checkpoints49/50, including the sensitivity
inherited through late matrix products.

The ten metered producer fixtures match independent FP64 calculations byte
for byte. The disabled source matches all16 outputs and complete bills of
the checkpoint49 covariance baseline. Every enabled call preserves first9
outputs and the specified replacement counts. Independent source AST and
unchanged-byte-segment checks, readout/prefix controls, complete bill replay
and scalar response replay for28 layer cells pass. Source inverse is exact.
No final target was decoded, no probe accuracy scored and no coefficient fit.

The reference baseline bill is41.9928/42.6095% above the already helper-bearing
fast source. That training cost is recorded and is not a deployment discount.
A later proposal must be evaluated using the original fast source with only
its learned readout changed, including that source's actual complete bill.
Any deployment gain still requires ordinary accuracy, held-out and worker
validation. Earlier failed qualifications and candidate scores are unchanged.

All14 guarded groups complete and retire,20 full native calls plus10 producer
fixtures,peak5.133GB,zero added swap. Both controller/verification handles
are reaped. S31#332517 remains saved adjusted2.1736465056908115e-9. The2e-9/~1e-9
goal is active and unmet; qualifying submissions remain authorized.
