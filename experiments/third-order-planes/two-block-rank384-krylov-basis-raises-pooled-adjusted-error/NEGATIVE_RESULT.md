# Fixed two-block Krylov384: closed negative

The192+192 basis retains two weighted-Gram Krylov blocks and a final
joint QR at rank384. Only the last shared-push basis block and inert
helpers changed; age4, original entry metric, physical factors, Lazy
schedule, learned literals and all other parent bytes are unchanged.

| Rank | Lazy raw MSE | Krylov raw MSE | Raw ratio | Own-cold adjusted ratio |
| --- | ---: | ---: | ---: | ---: |
| 35 | 1.870602849007794e-08 | 1.93089673284954e-08 | 1.0322323276 | 1.05057791591 |
| 100 | 1.842453417527493e-08 | 1.869881698217136e-08 | 1.01488682451 | 1.03257950602 |

Pooled adjusted ratio: **1.041615262270436**.
Both raw and adjusted scores worsen. The fixed both-wins and pooled
<=.99 gate fails. These are two reused ordinary1B development
networks, not independent generalization evidence or official scores.

Rank35 complete cold FLOPs: 398080917697 -> 405155903081 (+1.777273%).
Rank100 complete cold FLOPs: 400893063472 -> 407881895254 (+1.743316%).

The leading operator multiply count alone understated the measured
bill increase: block shapes, padding/kernel schedules, reorthogonalization,
QR and changed network geometry all remain paid. No selective bill reuse.

Four actual-kernel algebra cases pass against explicit FP64 dense Grams:
joining-only, joining plus old core, an FP32-rounded rank32 construction,
and an exactly zero operator. Rounding means the rank32 construction is
not claimed to have exact algebraic rank32. Zero exercises exact degeneracy.
Maximum relative operator error: 4.564946173133269e-06.
Maximum normalized orthogonality residual: 2.708021508247332e-08.
Krylov-block inclusion, input bytes, finite FP32 output, six malformed-input
rejections, repeated arrays and fresh-module bills pass. The original
unit harness stopped before a basis because its parent scratch arena
was not initialized. That exit1 and source are preserved; additive
unit_v2 fixes only arena setup/result location, with candidate and gates
unchanged. See HARNESS_CORRECTION.md.

Four full predictions pass exact own repeat, warm/cold100 equality,
first5 parent-layer bytes, finite nonnegative outputs and cache retirement.
Own predictions and bills were sealed before opening pinned truth arrays.
Total full-call charge: 1593873054858 FLOPs.
Peak RSS including self-reports: 6407585792 bytes; zero added swap.
All five groups retired: four successful stages and the preserved setup
failure. No numerical or upload worker remains.

This construction is closed without rank/split/iteration/seed/layer/metric/
QR/precision/head/blending/fit search. No wider panel, production client,
package or submission follows this failed candidate. The result is not
a general negative theorem about Krylov methods or alternative metrics.

General method background: [Musco and Musco2015](https://arxiv.org/abs/1504.05477).
No approximation theorem from that paper is transferred to this fixed
FP32 nonlinear estimator or its endpoint score.

Official bank stays Lazy #331599 adjusted 3.816449265779503e-09.
The4e-9 milestone is achieved;3e-9,2e-9 and approximately1e-9 remain unmet.
The goal stays active. Future qualified Phase2 uploads remain preauthorized.
