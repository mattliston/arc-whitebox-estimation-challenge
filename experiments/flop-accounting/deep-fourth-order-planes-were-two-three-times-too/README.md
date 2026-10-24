# Deep fourth-order planes were two to three times too small and overlapped the fitted correction

2026-09-04 · experiment · informational · measured

fourth-cumulant two-pair plane (K22) pre-plane card/truth 0.92 (L2) -> 0.41 (L14), corr 0.96-0.99; fourth-cumulant triple-single plane (K31) 0.52 -> 0.30; universal r35/r100 (FOURTH_PLANE_RATIOS.json). Forcing the 8M truth planes at the card's consumption point (RETRO_FORCE_PLANES): host fitted transport-energy residual correction (F2J)-off 8.62e-8 -> 2.08e-8 (r35), 7.70e-8 -> 2.97e-8 (r100); the engine's matrix-matrix multiplication (GEMM)-class self-pushed planes: 3.32e-8 / 3.79e-8 (its one 1024^4 term is worthless: NOBIGPAIR identical). Engine with the CARD's planes: k3 12.9% at L14 (worse than the card); with truth planes 6.0% at L12.

Recorded measurement:

    K22 pre-plane card/truth 0.92 (L2) -> 0.41 (L14), corr 0.96-0.99; K31 0.52 -> 0.30; universal r35/r100 (FOURTH_PLANE_RATIOS.json). Forcing the 8M truth planes at the card's consumption point (RETRO_FORCE_PLANES): host F2J-off 8.62e-8 -> 2.08e-8 (r35), 7.70e-8 -> 2.97e-8 (r100); the engine's GEMM-class self-pushed planes: 3.32e-8 / 3.79e-8 (its one 1024^4 term is worthless: NOBIGPAIR identical). Engine with the CARD's planes: k3 12.9% at L14 (worse than the card); with truth planes 6.0% at L12. The card's k3 with truth planes = the engine's (4.4-6.5%). Naive frozen scale tables explode the card (co-adapted quartic star); the frozen k4/k3 tables over-correct with right planes. BOARD: the effect is 84% inside the deployed operator's span (corr +0.91), perp ceiling 0.4% / 0.1%; corr(truth - deployed, truth - truth-plane host) = +0.90 -> the operator IS a learned plane fix; a plane chain + refit lands near the deployed card. Value: the derived (no-learning) replacement of the operator, and the host on which N0.1d/N0.1e become the whole error. T-75..T-77d: split into emission (fixed, certified) and push (the born-content wall). In-card plane ladder (r35 host): engine K22 alone 3.10e-8, + truth K31 2.47e-8, truth K22 + engine K31 2.86e-8, truth both 2.08e-8, + feeds 1.70e-8. T-78: sampled planes (N forwards + rank-8 projection) closed - at 65k forwards (the whole budget) the sampled K22 only equals the engine chain (23-25% error), K31 useless. T-79: the CP3 rank is not reducible even with truth planes (cap 8192 doubles the plane host's error); F32_COV free (-4.9% util). T-79d: option (a) bounded - the operator's own direction adds <= 2.4% on the engine-K22 plane host; a refit must find new tape directions worth > 20% just to tie -> adjusted ~0.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/fourth_plane_ratio_table.py](code/public_g29_bond_kernel_v1/fourth_plane_ratio_table.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L698](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L698)
- [FOURTH_PLANE_RATIOS.json#L1](FOURTH_PLANE_RATIOS.json#L1)
- [PERP_JUDGE_G29.json#L1](PERP_JUDGE_G29.json#L1)
- [inject_planes_beta_r35.log#L1](inject_planes_beta_r35.log#L1)
- [inject_planes_truth_k4tab_r35.log#L1](inject_planes_truth_k4tab_r35.log#L1)
- [inject_planes_r3fix_r35.log#L1](inject_planes_r3fix_r35.log#L1)
- [inject_planes_engine_r35.log#L1](inject_planes_engine_r35.log#L1)
- [inject_planes_beta_k4tab_r35.log#L1](inject_planes_beta_k4tab_r35.log#L1)
- [inject_planes_mix0.1_r35.log#L1](inject_planes_mix0.1_r35.log#L1)
- [inject_planes_truth_r100.log#L1](inject_planes_truth_r100.log#L1)
- [inject_planes_beta_k4tab_r100.log#L1](inject_planes_beta_k4tab_r100.log#L1)
- [inject_planes_engine_r100.log#L1](inject_planes_engine_r100.log#L1)
- [inject_planes_beta_r100.log#L1](inject_planes_beta_r100.log#L1)
- [inject_planes_mix0.3_r35.log#L1](inject_planes_mix0.3_r35.log#L1)
- [inject_planes_truth_r35.log#L1](inject_planes_truth_r35.log#L1)

<details>
<summary>All result files and records (15)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [FOURTH_PLANE_RATIOS.json](FOURTH_PLANE_RATIOS.json) · result
- [PERP_JUDGE_G29.json](PERP_JUDGE_G29.json) · result
- [inject_planes_beta_r35.log](inject_planes_beta_r35.log) · log
- [inject_planes_truth_k4tab_r35.log](inject_planes_truth_k4tab_r35.log) · log
- [inject_planes_r3fix_r35.log](inject_planes_r3fix_r35.log) · log
- [inject_planes_engine_r35.log](inject_planes_engine_r35.log) · log
- [inject_planes_beta_k4tab_r35.log](inject_planes_beta_k4tab_r35.log) · log
- [inject_planes_mix0.1_r35.log](inject_planes_mix0.1_r35.log) · log
- [inject_planes_truth_r100.log](inject_planes_truth_r100.log) · log
- [inject_planes_beta_k4tab_r100.log](inject_planes_beta_k4tab_r100.log) · log
- [inject_planes_engine_r100.log](inject_planes_engine_r100.log) · log
- [inject_planes_beta_r100.log](inject_planes_beta_r100.log) · log
- [inject_planes_mix0.3_r35.log](inject_planes_mix0.3_r35.log) · log
- [inject_planes_truth_r35.log](inject_planes_truth_r35.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-deep-fourth-order-planes-were-two-three-times-too) · [Copy provenance](../../../PROVENANCE.json)

