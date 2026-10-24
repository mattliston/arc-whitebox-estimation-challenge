# Full-matrix norm-coupling law lowers MSE 6.09% / 17.06% but needs about twelve cubic products per layer

2026-10-02T10:44Z · finding · informational · measured

The complete one-layer law run on the card's own covariance, means and third-order plane, with the conversion from the pushed bank columns, starting at zero at layer 0 and used only as the (3,1) fourth-cumulant plane (K31) extra plane, holds 0.76-0.88 of the oracle's network-specific energy and gives -6.09% on network 35 (oracle -7.99%) and -17.06% on held-out 651 (oracle -15.60%) with the law as first shipped (-7.30% on 35 corrected). Port pricing on 35: the lean law (no block product with (2,1) third-cumulant plane (K21)) costs 0.2 points; a state cut to rank 16 / 32 / 48 keeps -4.80 / -5.05 / -5.35% of -6.37%; a late start loses most of the final-layer gain (from layer 4: -2.93%, from layer 6: -1.57%); the research form needs about a dozen n^3 products a layer plus one over all bank columns, so it is a reference, not a port.

Recorded measurement:

    free run on true inputs holds 0.83 / 0.91 / 0.91 / 0.88 / 0.80 at layers 2 / 5 / 9 / 13 / 15 (without conversion 0.64-0.75); in the card: share 0.76 (layer 2), 0.82-0.88 (3-11), 0.84 / 0.80 (13 / 14); 35: conversion from bank columns -6.09% (all layers -8.72%), + b mode -5.69%, rank 32 -5.74%, strength 0.75 -6.37%, without conversion -2.08%; 651: strength 1 12.9566e-9 = -17.06% (all layers -9.04%), 0.75 -15.25%; value vs fidelity: one step from truth -8.1% / -14.0%, two steps -5.2% / -12.3%, old free run +3.3% / -2.6%; pricing on 35 at 0.75: complete -6.57%, lean -6.37%, rank 48 / 32 / 16 -5.35 / -5.05 / -4.80%, plane with only the multiple of C removed -5.42%, start layer 4 -2.93%, layer 6 -1.57%

## Programs

- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py)
- [../2026-10-02-gaussian-linear-response-collective-tilt-explains-88-97-norm/nlaw.py](../2026-10-02-gaussian-linear-response-collective-tilt-explains-88-97-norm/nlaw.py)
- [t19q_ablate.py](t19q_ablate.py)
- [t19r_fresh_rank.py](t19r_fresh_rank.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L244](../../TEAM_LOG_2026-10-01_to_10-22.md#L244)
- [TEAM_LOG_2026-10-01_to_10-22.md#L245](../../TEAM_LOG_2026-10-01_to_10-22.md#L245)
- [TEAM_LOG_2026-10-01_to_10-22.md#L250](../../TEAM_LOG_2026-10-01_to_10-22.md#L250)
- [TEAM_LOG_2026-10-01_to_10-22.md#L258](../../TEAM_LOG_2026-10-01_to_10-22.md#L258)
- [TEAM_LOG_2026-10-01_to_10-22.md#L263](../../TEAM_LOG_2026-10-01_to_10-22.md#L263)
- [TEAM_LOG_2026-10-01_to_10-22.md#L282](../../TEAM_LOG_2026-10-01_to_10-22.md#L282)
- [RESULT.md#L1166](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md#L1166)

<details>
<summary>All result files and records (4)</summary>

- [t4_ora10.log](t4_ora10.log) · log
- [t4_ora11.log](t4_ora11.log) · log
- [RESULT.md](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-full-matrix-norm-coupling-law-lowers-mse-6-09) · [Copy provenance](../../../PROVENANCE.json)

