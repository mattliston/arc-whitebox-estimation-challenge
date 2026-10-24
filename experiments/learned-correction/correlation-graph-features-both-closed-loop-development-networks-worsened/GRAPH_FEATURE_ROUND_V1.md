# Current full-pair correlation-graph round, version 1

Written September 4, 2026 PDT, after repaired-bank submission 329851.
Phase 2 only. One fixed 160/40, 40-epoch, C64 training round; no sweep.

The four-mode Gaussian triangle source is closed: unlearned final raw MSE
ratio versus the unchanged full-pair host is 0.998842 on development net 35
and 1.000108 on net 100 (independent organizer 1e9-frame means). The algebra
certificate passes, but this fixed approximation provides no useful gain.
No larger-rank run, strength fit, candidate port, or panel is justified.

Next hypothesis: correlation-neighborhood information predicts part of the
new full-pair card's remaining local error. This is a matched re-examination
of the modest honest T-137 graph result on a substantially changed card,
not another capacity or DAgger sweep. The current C64 mean head removes
23.49% of the held-out final one-step host error: 7.4542e-9 -> 5.7035e-9.

Append a fixed 20-column graph block to the current 30 features:
signed/absolute top-8 correlation-neighbor sums of alpha/gate/pdf (6);
top-8 squared correlation sum and largest absolute correlation (2);
softmax correlation-squared neighbor means at gamma 8 and 32 (6);
post-covariance eight-mode power-subspace loading, mean/rms correlation,
centered log variance, and covariance contractions against alpha/gate (6).
The subspace is defined by eight block-power iterations from a fixed DCT
seed, not a full eigendecomposition or an assertion of exact top modes.

Features are evaluated from host states before this layer's correction.
Generate all 200 rows afresh and require byte-identical host, sigma, and
original 30 features versus frozen parent rows. Check the late-layer
one-step profile and finite features before fitting. Preserve all inputs.
Same fixed split, seed, model width, optimizer, and 40-epoch schedule as the
parent; no target-fed post-state feature or leaked covariance is allowed.

Initial gate: compare the matched full-final-feature closed loops on nets
35 and 100 with the frozen full-final-feature parent. Require at least 5%
raw improvement on each before spending on scalar-final refitting and
metered packaging. Official release still requires the independent 14-net
panel, runtime/residual/memory checks, and substantial adjusted improvement
on both panels. A one-step training gain alone is not success.
