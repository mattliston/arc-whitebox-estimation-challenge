# Supply-quality oracle comparisons isolated plane, fourth-order and univariate inputs

2026-09-04 · experiment · rejected · measured

Teammate B (fork 'supply-ladder'): on the deploy-frame engine (r35) make one input exact per arm - truth fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31) (ORACLE_FOURTH), truth univariate k3, truth pre-third-cumulant repeated-index plane (K21) plane, all three - and grade each at optimal beta_v (and the k3/k4 arms at their optima). Current rung: +10.6 at beta 0.40 (60% noise; plane ~7% accurate; carried fourth-order planes 17-24% error). Ceiling: exact v = -54% raw.

Code availability: This tracker entry proposes oracle-comparison arms and quotes earlier measurements. The shared tensor engine and consumption instrument were recovered; the entry does not certify completion of every proposed arm.

Recorded measurement:

    Teammate B (fork 'supply-ladder'): on the deploy-frame engine (r35) make one input exact per arm - truth K22/K31 (ORACLE_FOURTH), truth univariate k3, truth pre-K21 plane, all three - and grade each at optimal beta_v (and the k3/k4 arms at their optima). Current rung: +10.6 at beta 0.40 (60% noise; plane ~7% accurate; carried fourth-order planes 17-24% error). Ceiling: exact v = -54% raw. Prices mechanism leaves N1.2c/d/e (P2R0G19B_MECH_N1_2C_CARRY, _N1_2D_FOURTH, _N1_2E_UNIVARIATE). RESULT (T-20 parts 1+2, teammate B; receipts experiments/p2_width1024_depth16/public_g19_attribution_v1/ladder_*.log, dvhat_rank0035_dep_*.npz, dvhat_rank0035_true_pairs.npz): current engine cuts 10.6% (beta 0.40); truth k4u+K22/K31 -> 7.4 (k4 arm +5.7); truth k3 -> 7.9 (k3 arm +6.8); truth pre-plane -> 7.8; all three -> 7.5 with identical per-layer dv fidelity; TRUTH pre-planes through the tangent -> 34.8 (53.3 joint); truth STATES 47.4 / 73.5. Wedges: gate-only carry 7.9 (CP-preserving carry keeps ~3/4); truth post-plane slices 8.1 (emission not the limiter); card-plane arm void (transcription). Transport is not the loss; collective mode negligible; per-layer beta vectors do not transfer. Closed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py](code/p2_width1024_depth16/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py](code/p2_width1024_depth16/public_g19_attribution_v1/consumption_betas.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [LANE_g3_tracker.md#L458](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L458)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-supply-quality-oracle-comparisons-isolated-plane-fourth-order-univariate) · [Copy provenance](../../../PROVENANCE.json)

