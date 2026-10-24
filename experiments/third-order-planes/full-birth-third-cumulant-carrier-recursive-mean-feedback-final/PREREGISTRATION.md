# Phase-2 CP3 full-birth cap ladder: target-free preregistration

## Purpose

Measure whether the full Phase-2 FLOP budget can carry substantially more of
the analytic third-cumulant state than the score-floor cap-1,792 host.  Raw
MSE, rather than adjusted-score multiplier, is the scientific objective.  No
accuracy result may be opened before a separate independent audit authorizes
it.

## Frozen family

One standalone source defines caps 2,048, 4,096, 8,192, and 16,384.  The four
classes share every operation except the integer CP cap.  They retain:

- float64 Hermite-10 mean/covariance host;
- float32 `lambda * sym3(p,p,q)` cumulant state;
- exact affine factor push and diagonal K3 read;
- inherited ReLU gate and leading Gaussian-star birth;
- target-blind factor-energy truncation when the post-birth rank exceeds cap;
- open-loop analytic recurrence.

At width 1,024 and depth 16, births occur only after h1 through h15.  The
rank-16,384 class therefore never truncates: its h1..h15 ranks are
1,024, 2,048, ..., 15,360, and W16 acts on rank 15,360.

## Algebraic resource screen

flopscope 0.12 charges a `(1024,1024) @ (1024,r)` matrix product
`2 * 1024^2 * r` FLOPs.  Each layer after the first pushes both P and Q, so
the CP affine-action charge is `4 * 1024^2 * sum(r_before_layer)`.

| cap | sum of acted-on ranks | CP affine-action FLOPs |
|---:|---:|---:|
| 2,048 | 29,696 | 124,554,051,584 |
| 4,096 | 55,296 | 231,928,233,984 |
| 8,192 | 94,208 | 395,136,991,232 |
| 16,384 | 122,880 | 515,396,075,520 |

Actual total FLOPs also include the shared covariance host, K3 reads, births,
and any truncation; retained whestbench 0.16 / flopscope 0.12 receipts govern.

The no-truncation terminal P and Q state has
`2 * 1024 * 15360 * 4 = 125,829,120` bytes of float32 factor payload, plus
lambda, weights, analytic state, temporaries, and process overhead.  Actual
sampled solution-process RSS must remain below 8 GiB.

## Resource gates

- total measured FLOPs no greater than the full `2^41 = 2,199,023,255,552`;
- sampled solution-process RSS below 8 GiB;
- residual time no greater than 0.4 seconds and wall time no greater than 120
  seconds;
- finite float32 output with shape `(16, 1024)`;
- synthetic 1,000-sample plumbing truth is scientifically unscored.

## TT boundary

No TT is included in this ladder.  A TT constructed solely to represent one
dropped CP block for one affine action is not a distinct capacity hypothesis
and is less efficient than retaining that CP block exactly.  A future TT rung
must propagate a residual for multiple layers, specify a target-free rounding
law, and beat a matched-FLOP CP cap to be scientifically distinguishable.
