# Fourth-order plane repairs overlap the learned residual correction

2026-09-02 · lesson · informational · measured

Replacing inaccurate fourth-order planes greatly reduced the uncorrected host error but largely reproduced a correction already learned from the observation tape. A candidate must improve the residual component outside the existing learned correction, rather than only the host error.

Recorded measurement:

    Receipts: LANE_g3_tracker.md T-71..T-73; public_g29_bond_kernel_v1 (FOURTH_PLANE_RATIOS.json, fourth_plane_scale_table.json, PERP_JUDGE_G29.json, inject_planes_*_r{35,100}.log, engine_ablation_table_r35.txt); estimator_retro.py RETRO_FORCE_PLANES / RETRO_PLANE_SCALE hooks; run_tensor_recursion_v4.py HOST_FOURTH / NOBIGPAIR.
    
    - Mechanism: the card's CP4 (rank 2048) + quartic-star machinery under-scales the fourth-order PRE-PLANES K22/K31 by 2-3× at depth (same disease as the univariate k4, universal, near-pure scale for K22). Those planes feed the pair series' Edgeworth weights → the skew (K21/k3) emission → everything downstream. Engine with the card's planes: k3 error 12.9% at L14 (worse than the card's 11.1%); with its own GEMM-class self-pushed planes 7.3%; with truth planes 6.0% at L12.
    - The engine's plane chain (`fourth_emission_1024` + `fourth_push_1024`) is deployable: its only 1024⁴ term (the K22 pair transport) contributes nothing (NOBIGPAIR identical to 4 digits) and the truth K22 plane is 96-98% rank-8 anyway; ~16 GEMMs/layer (~22% util, replacing the CP4's ~11%).
    - Naively SCALING the card's assembled pre-planes by the frozen table explodes the card (1e-4) — the quartic star / pairing_alpha assembly is co-adapted (closure-coadaptation-basin-law); a consistent REPLACEMENT (truth or engine planes) does not explode. The frozen k4/k3 tables were fitted against the under-scaled planes and over-correct once the planes are right (1.4-1.7e-7): drop them with corrected planes.
    - Board frame: the plane effect is what the deployed operator already does (84% in-span, perp ceiling 0.4%/0.1%); corr(truth − deployed, truth − truth-plane host) = +0.90. Perpendicular content given planes: truth k3 → host 1.08e-8 (judge +0.43, ceiling 19%), truth v → 1.27e-8 (+0.35, 12%), both → 2.2e-9 (+0.56, 31%); the feed +0.27 (7%).
    
    - Plane-chain fault split (T-75..T-77, one-step certificates with truth POST planes from mc_post_planes_mps.py): the EMISSION is right once E[φ³] carries the same Edgeworth corrections as the other moments (FOURTH_R3FIX=1: K31post corr 0.33 → 0.998; = teammate A's T-49 g24fix); the PUSH is the fault: the plane-supported push drops the post tensor's (2,1,1)/(1,1,1,1) classes → K22 25-32% missing, K31 ~95% missing, k4 28-37% (the k4 table's source). The missing part is uncorrelated with the pushed planes (not a rescale), 52-65% rank-8. In-card plane ladder (r35 host): engine K22 alone 3.10e-8, +truth K31 2.47e-8, truth both 2.08e-8 — K22 and K31 accuracy worth ~0.6-0.8e-8 each.
    
    **Why:** explains the 69% learned correction physically (Matt: "derive everything"), and reframes N0.1: the leftover after the operator is the (k3, v) accumulated residual GIVEN correct planes — the engine's own 6% third-order transport error is the next mechanism to name.
    
    **How to apply:** judge every candidate by the perpendicular judge BEFORE any refit; a derived plane chain is the no-learning replacement of the operator (worth ~engine-class 3.3-3.8e-8 F2J-off), not a board gain; the road to 1e-8 = planes + k3-given-planes (engine truncations: second-order Edgeworth, fourth-order insertions, (2,1,1) feed) + v. Related: [[plane-feed-mechanism-live]], [[operator-span-redundancy-law]], [[card-kurtosis-cap-defect]], [[closure-coadaptation-basin-law]].

## Programs

- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [estimator_retro.py](estimator_retro.py)
- [mechfeat15.py](mechfeat15.py)
- [k22_chain.py](k22_chain.py)
- [mc_post_planes_mps.py](mc_post_planes_mps.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)

[Research graph](../../../graph/index.html#node=2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction) · [Copy provenance](../../../PROVENANCE.json)

