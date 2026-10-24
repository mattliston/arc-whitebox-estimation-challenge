# One-layer centering audit of the existing joint-moment supply

Phase 2 only, September 5, 2026, 17:09 PDT. The planned current pair oracle
stopped at rank35 L8 BEFORE any wide attribution: stored pre K22/K31
diagonals differ from coherent8M marginal k4 by 1.484% / 1.202%, above the
script's 1% consistency check. Source and empty v1 folder are preserved.
Read-only checks at L2/5/8/9/12/14/15 show 1.1-2.9% fourth discrepancy,
but only .055-.090% K3 discrepancy. Numerical cancellation is a hypothesis,
not an established cause or a reason to relax the check.

One fixed measurement: rank35 L12, SAME original 8M stream, batch65536
(the old plane producer's batch), CPU PCG64DXSM and MPS float32 network.
Accumulate both the old raw pair powers and powers after subtracting the
stored coherent8M pre mean BEFORE products. For each, CPUfloat64 batch
accumulation and complete recentering of the residual mean. Also collect
same-stream centered post covariance and mean. No independent seed or
sample-size sweep; this is an arithmetic audit of different moment data,
not reopening the already-completed covariance replica diagnostic.

First certify the shift/recentering algebra on a small CPUfloat64 random
dataset. Then compare regenerated raw planes against archived planes;
compare centered plane diagonals to independently summed centered marginal
moments. Compare the raw-vs-centered plane difference AFTER the current
Gauss6/Edge2 response, and against the current covariance innovation. Do
not infer materiality from unweighted K4 relative error alone.

If the two accumulation conventions differ by more than 5% of the current
innovation response norm, do not use old planes for attribution; acquire
only the fixed L8..14 centered supplies needed for the already-defined
35/100 diagnostic. Otherwise preserve the discrepancy and use explicit
consumer-weighted numerical qualification, not a claimed perfect oracle.
No fit, payload, candidate, raw-score, panel or submission follows from
this data check alone. One nice10 MPS job, torchthreads4, no threadpoolctl.
