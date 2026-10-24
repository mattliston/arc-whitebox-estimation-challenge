# Canonical fourth-trace projection — CLOSED

2026-09-06 03:45:37 PDT, Phase 2. The canonical Euclidean harmonic
projection was independently certified against explicit symmetric tensors:
adjoint/trace/residual orthogonality errors<=2.67e-15; Pythagoras<=2.85e-14.
This is genuinely the orthogonal trace-generated component, unlike the
previous covariance-metric trace-matching reconstruction.

No new samples. Same rank35 post12->pre13->mean response14, two existing
independent8M statistic files, same actual bank frozen inputs. All six
target response arrays are byte-identical to the prior oracle audit.
Core reconstruction6.18e-17 relative, direct pair response3.22e-15 absolute.

| Consumer | Residual/baseline signal cross-energy | Cross A->B | Cross B->A |
|---|---:|---:|---:|
| Covariance response | 0.6336143110 | 0.6454972814 | 0.6434175949 |
| Next mean response | 0.6108077062 | 0.6269336997 | 0.6193154809 |

Both fail required signal ratio<=0.50, despite ordinary cross ratios<=0.65.
Signal/noise remains42.73/39.37. These are isolated fourth-contribution
response comparisons, NOT total error or an adjusted/raw estimator score.
The canonical projection's orthogonality does not imply downstream
value optimality under these nonlinear, weight-dependent consumers.

No metric/source/scale/rank/layer/count/seed sweep, sampling expansion,
carrier, fit, port, panel or submission. Both tested full-trace
reconstructions are closed under their designs; not a proof that every
fourth-order representation is useless. Complete full-carrier cost
remains unfavorable and untested in wide accuracy.

Receipt canonical_fourth_trace_value_v1.json/.npz. Wrapper SHA256
917a70060e0ac267b2c329b918e5a93e13257ecf51abd723f958df1d4d661632.
Canonical helper SHA256
6865a852ee629e3712020e6b654379418a2241ebc1b95dff590a6e4b40e9f0fc.
