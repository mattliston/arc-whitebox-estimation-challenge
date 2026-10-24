# Frozen skew-scale table recovered 34–53% of the true-skew correction value

2026-09-04 · experiment · informational · measured

Per-unit k3 recursion = the diagonal of the same third-order object as N1.2; the engine's k3 supply is worthless at any beta on the deploy-frame run (best -0.8). Re-grade after each N1.2 fix (SUB-4). HONEST FRAME (T-36, independent 4M truth, r35, fitted transport-energy residual correction (F2J)-on, deployed 2.704e-8, +-3 points): truth per-unit k3 fed into the card's EXISTING univariate mean map cuts 30.2% (24.5 from layers 8-15, 7.6 from 2-7); the engine's carried per-unit k3 (deploy frame) cuts 14.7% (BK3 0.33) on its own; with truth univariate k3 fed to the engine its k3 arm reaches 29.6.

Code availability: The skew-table module and shared retrofit instrument were recovered. The tracker aggregates several forcing and engine-supply arms; a dedicated driver for this exact aggregate comparison was not identified.

Recorded measurement:

    Per-unit k3 recursion = the diagonal of the same third-order object as N1.2; the engine's k3 supply is worthless at any beta on the deploy-frame run (best -0.8). Re-grade after each N1.2 fix (SUB-4). HONEST FRAME (T-36, independent 4M truth, r35, F2J-on, deployed 2.704e-8, +-3 points): truth per-unit k3 fed into the card's EXISTING univariate mean map cuts 30.2% (24.5 from layers 8-15, 7.6 from 2-7); the engine's carried per-unit k3 (deploy frame) cuts 14.7% (BK3 0.33) on its own; with truth univariate k3 fed to the engine its k3 arm reaches 29.6. Consumer cost zero. The card's k3 error ~6% of k3 is 3-14x the truth's k3 noise from L5. Task (teammate A): supply a better 1024-vector k3 per deep layer - grade engine k3u, D's k3 features, and the coincidence-resummed generation (derived) under the independent truth. Receipts: experiments/p2_width1024_depth16/public_g21_pair_consumer_v1/LANE.md section 8, experiments/p2_width1024_depth16/public_g19_math_acceleration_v1/LANE_g3_tracker.md T-36/T-37. T-43: k3 (with k4) is the bulk of the in-card payoff on BOTH nets when forced jointly with v (the net-specific reading was the linear frame's). T-44: card k3 3.5->11% off vs the 32M truth; run the cum4 feed and the early-window residual carry TOGETHER with the k4 fix. T-48 task: cum4 (2,1,1) feed + early-window residual carry TOGETHER on top of the k4 rescale (teammate A), F2J off, independent scoring; target the structured-halving regime. T-55/T-56 (teammate A, public_g27_k3_structure_v1): card k3 mildly under-scaled (panel ratio 0.996 -> 0.929, sd <= 0.016); injecting Delta k3 = card_k3 * (1/s_l - 1) on top of the k4 table takes the host from 5.27e-8 -> 4.12e-8 (r35) and 5.00e-8 -> 3.84e-8 (r100), F2J off, independent 4M, u unchanged; truth-Delta k3 reference 1.91e-8 / 2.83e-8. Deep k3 error = ~30% scale (done) + ~0% regime (alpha/g3/v bins are noise; a LOO alpha table hurts) + ~70% unit-specific accumulated carry error (held-out ridge on tape features explains 10-12%; a learned corrector ~35% pooled, 0.5 at depth). T-52: the arena-relaxed full-depth residual carry buys only -9..-12% at +26-39% util; partial carries are worse than none. T-62/T-63: a learned in-card k3 corrector (features R^2 0.41-0.55 at depth) cuts the rescaled host 15-20% but lies inside the operator's span (perpendicular alignment -0.1..+0.1); mechanistic per-unit features lift held-out R^2 only marginally (0.135->0.147 at L10); the engine's k3u injected in-card does not beat the board. Unit-level k3 is now part of N0.1.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/k3_table.py](code/public_g20_emission_retrofit_v1/k3_table.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L616](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L616)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-frozen-skew-scale-table-recovered-3453-true-skew-correction) · [Copy provenance](../../../PROVENANCE.json)

