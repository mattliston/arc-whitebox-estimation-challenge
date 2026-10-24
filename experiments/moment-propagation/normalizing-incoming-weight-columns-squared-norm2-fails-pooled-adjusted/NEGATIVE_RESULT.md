# Incoming-column normalization: closed negative at fixed development gate

One fixed positive-diagonal gauge normalizes hidden incoming columns to
squared norm2, with final scale1. FP64 scale arithmetic and FP32 stored
weights; restore every output row. One final estimator-call replacement
plus inert helpers/proxy; all other parent bytes unchanged.

Exact network invariance follows from positive homogeneity in real
arithmetic. Stored-weight rounding was independently checked; approximate
estimator invariance is neither asserted nor required. The existing radial
projection has a legitimate original-coordinate interpretation; this is
an empirical coordinate choice, not a repair of a demonstrated bug.

| Rank | Lazy raw MSE | Normalized raw MSE | Raw ratio | Own-cold adjusted ratio |
| --- | ---: | ---: | ---: | ---: |
| 35 | 1.870602849007794e-08 | 1.865189844818023e-08 | 0.997106278229 | 0.997806269429 |
| 100 | 1.842453417527493e-08 | 1.855402639447829e-08 | 1.00702824929 | 1.00773024685 |

Pooled adjusted ratio **1.002748104451459**.
The gate requires raw AND adjusted wins on BOTH networks and pooled<=.99.
These are two reused ordinary1B development networks, not independent
generalization evidence or an official score. All transformation and
restoration work is included in the candidate complete cold bills.

Rank35 cold bill 398080917697->398360379521 (+0.070202%). First-layer maximum absolute difference 9.53674316406e-07 (limit2e-5).
Rank100 cold bill 400893063472->401172525296 (+0.069710%). First-layer maximum absolute difference 8.34465026855e-07 (limit2e-5).

Three actual-FNP synthetic direct-network cases, zero-column handling,
four malformed-input rejections, input preservation, exact repeated
arrays/bills and column-norm checks pass.
Maximum scaled direct-network output error 5.031942980904821e-08.
Four full calls pass own repeat, warm/cold100 bytes, restored first-layer
tolerance, finite nonnegative16x1024 output and original cache retirement.
Predictions and own bills were sealed before the two pinned truth files.

Total full-call charge 1566531889278F; peak RSS 6610763776B.
All four guarded stages exit0/retire with zero added swap. No numerical
or upload worker remains. All eight prospective source files are unchanged.

Close this fixed normalization without a scale/target/layer/order/
precision/kernel/head/metric/fit/blending search. No wider panel,
production client, archive or submission follows. No universal claim
about every positive scaling or jointly retrained estimator follows.

Official bank stays Lazy #331599 adjusted 3.816449265779503e-09.
The4e-9 milestone is achieved;3e-9,2e-9,~1e-9 remain active/unmet.
Standing user authorization covers future qualified Phase2 uploads.
