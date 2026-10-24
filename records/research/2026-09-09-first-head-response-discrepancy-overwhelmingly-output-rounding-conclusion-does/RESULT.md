# First-head discrepancy is overwhelmingly return rounding

Completed2026-09-09UTC. Existing-artifact diagnostic, Phase2 only.
Official bank330390 remains adjusted4.53886349950395e-9,
raw2.120838555441651e-8. No new score or submission.

All40 nativeN/CPU/nativeL/nativeR endpoint reconstructions passed exactly:
FP32(max(FP64host+ownedDelta,0)) equals saved served row1 bytes. All original
individual scalar values and secants replayed exactly. No network, gradient,
model fit, weight-array decode or truth-array read occurred. The only122
arrays decoded were existing small outputs/hosts/deltas/cotangents.

Primary D is nativeN minus CPU's original fixed layer1 response. Decomposing
D into pre-return difference plus differential FP32 return rounding gives:

|Rank|Original h|Pre-return/served vector norm|Absolute pre-return/served scalar|Signed rounding fraction|
|---|---:|---:|---:|---:|
|35|.025|.000342698868|.000373578936|.999626421064|
|35|.05|.000709490273|.000756660329|.999243339671|
|100|.025|.000493315515|.000208578605|.999791421395|
|100|.05|.000971049682|.002173446011|.997826553989|

Every primary denominator passed the predeclared nontrivial floors and both
ratios are below the fixed.25 local-dominance threshold. Vector/scalar
additive-identity residuals are exactly zero. Return rounding explains
99.78–99.98% of the SIGNED scalar discrepancy on these four cases; pre-return
vector discrepancies are only8.21e-9,7.91e-9,1.02e-8,1.00e-8 in L2 norm.
These are derivative-response diagnostics, not raw prediction MSE.

The secondary N-versus-L/R responses show the same broad pattern. The
rank100,h=.05 N-L scalar is only2.2734672e-11, below the original1e-10
nontrivial floor, and stays INCONCLUSIVE rather than being ratio-promoted.
No L/R circuit is selected. The original CB covariance attribution gate
remains FAILED; its observed served-response changes predominantly move
the final quantization outcome in this local test, rather than demonstrating
a large smooth first-head response change. This is a local interpretation,
not an assertion that covariance rounding never affects later layers.

## What remains unproved

These captures stop at layer1. The whole16-layer BR/NG scalar failures are
NOT retrospectively qualified. No current native derivative is certified;
no optimizer is resumed. The fixed ND2 proposal remains a completed failed
transfer result. This evidence supports considering a separately specified
all16 pre-return capture, not lowering the old5% check or changing its inputs.
No new raw gain, bank improvement, complete inference bill or runtime change.

## Execution and provenance

Worker4.7610424580s, lifetimepeakRSS360775680B. Independent whole watcher
5.1991585009s, maximum SAMPLED combinedRSS382877696B, maximum sample
interval.2515839580s, zero swap growth, no forced termination. Exit0;
exec89179 finished, return_quantization_session=null. No numerical worker
left by watcher. PASS means completed diagnostic/integrity/resource checks.

Both result/completion pairs were verified. Root independently rehashed
all810 distinct recorded analysis pins plus watcher sources/artifacts after
completion without re-decoding arrays. All unchanged. Frozen files:

- Worker66597379e5aa9d86f6207440ee6f2db519e11b998ae8092930fba19904b53837.
- Protocolb8c47188aab2c49e1699ac73b8ac251c8ef26727947e87ea7e03919b0d4d26f0.
- Watcherf79749448c5a6015540474b18b58b6981944fc8b8c9182200effc0095728e32b.
- Analysis result759bc9902d388c6ad223742923955884294af4650dc76f083f4959614c1d4a39.
- Watch result468022dcb1dc5703ea5d96554805157b2a8644335ea66fd89843ed163725ace5.

Preserve this package and all old receipts; do not rerun or edit its sources.
Goal remains ACTIVE official<1e-9, banks<4e-9,<3e-9,<2e-9. The local4.48
confirmation request is already fulfilled by official330390; do not duplicate.
