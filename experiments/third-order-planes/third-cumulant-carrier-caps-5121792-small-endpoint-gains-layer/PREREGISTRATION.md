# Target-blind CP3 K3-only cap ladder

Status: frozen before any Phase 2 target opening. The analytic host and all CP
mechanics are identical across candidates; only the literal cap changes.

## Immutable ladder

The unconditional accuracy ladder is fixed to:

1. cap 512: the preserved source authority in
   `../cp3_cap512_k3_readout/estimator.py`;
2. cap 1024: `../cp3_cap1024_k3_readout/estimator.py`;
3. cap 1536: `../cp3_cap1536_k3_readout/estimator.py`.

No cap may be added, removed, or moved in response to target accuracy. All
three candidates use the same fixed Hermite-10 analytic recursion, exact CP
component-energy truncation, and coefficient-one open-loop K3 mean readout.
They contain no fit, checkpoint, teacher state, K21 matrix, K4, TT residual,
future-weight geometry, or PSD repair.

Cap 1792 is a meter-only conditional candidate in
`../cp3_cap1792_k3_readout/estimator.py`. It joins the sealed accuracy ladder
only if, before any target is opened, the common provenance-complete driver
confirms total FLOPs strictly below `0.1 * 2^41 = 219,902,325,555.2`, residual
time below 0.4 seconds, wall time below 120 seconds, finite `(16,1024)` output,
and memory below 8192 MiB. If any gate fails or cannot be authoritatively
measured, cap 1792 is excluded without consulting accuracy.

Cap 2048 is excluded from this ladder. Its static total already exceeds the
score floor; testing it later would require a separate, predeclared
compute-adjusted tradeoff experiment.

## Static cost result

`derive_static_ladder.py` verifies normalized-AST identity across all four
sources and enumerates the v0.12 CP operations. Using the provisional analytic
base only to project totals, the source arithmetic gives:

| Cap | Exact static CP increment | Projected total | Budget share | Floor headroom |
|---:|---:|---:|---:|---:|
| 512 | 32,538,849,280 | 136,462,170,112 | 6.2056% | 83.440B |
| 1024 | 64,888,818,688 | 168,812,139,520 | 7.6767% | 51.090B |
| 1536 | 95,095,655,424 | 199,018,976,256 | 9.0503% | 20.883B |
| 1792 | 110,199,020,544 | 214,122,341,376 | 9.7372% | 5.780B |

The cap-512 arithmetic differs from the earlier target-blind exploratory bill
by only two FLOPs. That run did not retain raw stdout, and the analytic base
receipt is stale, so the projected totals are not receipts. The common driver
must regenerate every admitted candidate and bind raw stdout/stderr, exact
commands, environment, source hashes, operation census, timing, and actual
memory evidence.

## Sealing and selection

All admitted candidates must produce and hash predictions before any target
metric is read. Once targets are opened, report raw final-layer MSE and the
official compute-adjusted score for every sealed cap. Select by locked
network-balanced final-layer compute-adjusted score; all-layer MSE is only a
diagnostic. No new cap or coefficient may be introduced after that comparison.
