# Kerdock quadrature and empirical-gate control variates adapted to Phase 2: endpoint accuracy failed

2026-08-26 · experiment · rejected · measured

The literal audit corrects the earlier method account: Phase 1 used a complete real-Kerdock MUB rule plus exact low-layer moment snaps, while #327723 used a nonlinear empirical-gate CV. Both were reconstructed for Phase 2. Every target-blind rank-0 form fails: Hermite D12 endpoint error is 7.9612e-8, the best degreewise snapped construction is 7.7496e-8 with wrong-signed truth correlation, and the empirical-gate replay remains far outside the endpoint gate.

Recorded measurement:

    The literal audit corrects the earlier method account: Phase 1 used a complete real-Kerdock MUB rule plus exact low-layer moment snaps, while #327723 used a nonlinear empirical-gate CV. Both were reconstructed for Phase 2. Every target-blind rank-0 form fails: Hermite D12 endpoint error is 7.9612e-8, the best degreewise snapped construction is 7.7496e-8 with wrong-signed truth correlation, and the empirical-gate replay remains far outside the endpoint gate.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0g1_kerdock_prefix_cost_v1/derive_cost_ledger.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0g1_kerdock_prefix_cost_v1/derive_cost_ledger.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/run_full_estimator_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/run_full_estimator_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/run_full_estimator_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/run_full_estimator_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exposure_trained_residual_state_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exposure_trained_residual_state_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/analyze_degree_shrink.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/analyze_degree_shrink.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/analyze_full_kerdock_basis_cv.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/analyze_full_kerdock_basis_cv.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/analyze_phase1_empirical_gate_cv.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/analyze_phase1_empirical_gate_cv.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/build_terminal_features.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/build_terminal_features.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/extract_native_q24_covariance.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/extract_native_q24_covariance.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/run_action_covariance_snap.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/run_action_covariance_snap.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/standard_real_kerdock.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_phase1_kerdock_hidden_snap_v1/standard_real_kerdock.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py)

</details>

## Evidence

- [COST_LEDGER.json#L1](COST_LEDGER.json#L1)
- [TERMINAL_RECEIPT.json#L1](../../third-order-planes/endpoint-action-hidden-state-compression-measured-rejected/TERMINAL_RECEIPT.json#L1)
- [FULL_ESTIMATOR_GATE_RESULT.json#L1](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/FULL_ESTIMATOR_GATE_RESULT.json#L1)
- [FULL_ESTIMATOR_GATE_RESULT-3722d52d.json#L1](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/FULL_ESTIMATOR_GATE_RESULT-3722d52d.json#L1)
- [STANDARD_REAL_KERDOCK_PROOF.json#L1](STANDARD_REAL_KERDOCK_PROOF.json#L1)
- [TERMINAL_RECEIPT.json#L1](TERMINAL_RECEIPT.json#L1)
- [RESULT.json#L1](RESULT.json#L1)

<details>
<summary>All result files and records (7)</summary>

- [COST_LEDGER.json](COST_LEDGER.json) · result
- [TERMINAL_RECEIPT.json](../../third-order-planes/endpoint-action-hidden-state-compression-measured-rejected/TERMINAL_RECEIPT.json) · result
- [FULL_ESTIMATOR_GATE_RESULT.json](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/FULL_ESTIMATOR_GATE_RESULT.json) · result
- [FULL_ESTIMATOR_GATE_RESULT-3722d52d.json](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/FULL_ESTIMATOR_GATE_RESULT-3722d52d.json) · result
- [STANDARD_REAL_KERDOCK_PROOF.json](STANDARD_REAL_KERDOCK_PROOF.json) · result
- [TERMINAL_RECEIPT.json](TERMINAL_RECEIPT.json) · result
- [RESULT.json](RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-kerdock-quadrature-empirical-gate-control-variates-adapted-phase-2) · [Copy provenance](../../../PROVENANCE.json)

