# Low-rank third-cumulant plane carry lost the distinct tensor content required by transport

2026-09-04 · experiment · rejected · measured

CLOSED (my error): a faithful carry emulation (pre_k21 = W^T K21_post W from the truncated post plane) deviates 400-600% from the bank's plane at every layer and the host explodes (2e-2): the third-order object is a TENSOR carried as CP factors (K21_ab = sum_r lam_r p_ra^2 q_rb); its push acts on the factors (p -> W^T p, q -> W^T q), not as a congruence of the (a,a,b) plane. The plane's low matrix rank is a consequence of the factors (p^2 weighting), not a cheaper carrier; the cost is set by the factor bank's rank (the youngest cohorts' Q are full rank, T-95b) -> exactly the T-95..T-97 verdict: compression closed. Both plane chains killed.

Recorded measurement:

    - T-119 THE THIRD-ORDER PLANE IS LOW-RANK (r35, from the bank dumps bankcp_r35_L*.npz, K21 = P diag(lam) Q^T): energy at rank 64 / 128 / 256: L4 82.7 / 95.9 / 99.7%; L8 95.0 / 99.4 / 100%; L12 98.2 / 99.9 / 100%; L14 98.8 / 99.9 / 100%, while the bank carrying it has rank 4096-14336 and the arena push costs 45% of the budget. A rank-128 PLANE carry (push (W^T U) S (W^T V)^T, add the births, re-truncate by a randomized SVD) would cost ~1% of the budget. T-95..T-97 closed the COHORT-bank compression (per-cohort factor ranks, dense reads); this is different: the plane itself. Necessary-condition test: RETRO_PLANE_RANK=r truncates the plane at every layer (numpy emulation, unbilled) -> honest loop raw at r=128/64.
    
    - T-119 first numbers (r35, un-adapted HF1 head, 8M): plane truncated at every layer to rank 128 -> 1.40e-7; rank 64 -> 3.08e-7 (loop alone 2.49e-8). As with every state change, meaningless before co-adaptation. COST MODEL of a real rank-r plane carry: push (W^T U) S (W^T V)^T = 2 x 1024^2 x r; births = the new cohort G (dense 1024x1024) added then re-truncated by a randomized SVD of a 1024 x (r+1024) matrix (~3e8 for r=128); read k3 = diag(U S V^T) trivial -> ~0.2-1% of the budget replacing the arena's 45% (the cubic star push 9% stays; with the fourth-order machinery also replaced by the closure, the card would sit near util 0.15-0.2, where the bar needs raw < ~5e-8 grader). Running chain_planerank.sh: plain host at rank 256/128, then co-adaptation at rank 128 (rows_meanleak_mech15_pr128 -> GNN_HPR128 -> loops).
    
    - T-119b BUILT: a faithful (unbilled numpy) emulation of a rank-r PLANE CARRY (RETRO_PLANE_CARRY=r): after each reset the post plane k21_target is truncated to rank r and CARRIED; at the next layer pre_k21 = W^T K21_post W replaces the bank read + cubic star, so truncation errors compound as in a real card; plane_carry_err_l dumps the relative deviation from the bank's plane. Co-adaptation chain queued behind the read-truncation chain.
    
    - T-119 CLOSED (my error): a faithful carry emulation (pre_k21 = W^T K21_post W from the truncated post plane) deviates 400-600% from the bank's plane at every layer and the host explodes (2e-2): the third-order object is a TENSOR carried as CP factors (K21_ab = sum_r lam_r p_ra^2 q_rb); its push acts on the factors (p -> W^T p, q -> W^T q), not as a congruence of the (a,a,b) plane. The plane's low matrix rank is a consequence of the factors (p^2 weighting), not a cheaper carrier; the cost is set by the factor bank's rank (the youngest cohorts' Q are full rank, T-95b) -> exactly the T-95..T-97 verdict: compression closed. Both plane chains killed. The read-truncation test is moot (no cheap representation produces the truncated plane).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/chain_planerank.sh](code/public_g22_operator_features_v1/chain_planerank.sh)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15.py)
- [code/public_g22_operator_features_v1/score_vs_truthpost.py](code/public_g22_operator_features_v1/score_vs_truthpost.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L901](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L901)
- [LANE_g3_tracker.md#L902](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L902)
- [LANE_g3_tracker.md#L903](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L903)
- [LANE_g3_tracker.md#L904](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L904)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-low-rank-third-cumulant-plane-carry-lost-distinct-tensor) · [Copy provenance](../../../PROVENANCE.json)

