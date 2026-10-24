# Deployed-estimator score economics at the minimum charged utilisation: verified

2026-08-26 · experiment · informational · measured

From exact board #328586, adjusted/raw is 0.7228361387704203. Beating 7.70e-9 at unchanged raw 2.9409214512554626e-8 would require multiplier <0.2618227017492396 and constant-cost equivalent <=575,754,209,978 FLOPs. Conversely raw <1e-8 at the current multiplier projects adjusted <7.2284e-9.

Recorded measurement:

    From exact board #328586, adjusted/raw is 0.7228361387704203. Beating 7.70e-9 at unchanged raw 2.9409214512554626e-8 would require multiplier <0.2618227017492396 and constant-cost equivalent <=575,754,209,978 FLOPs. Conversely raw <1e-8 at the current multiplier projects adjusted <7.2284e-9.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/verify_tiny.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/verify_tiny.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/run_workspace_v4_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/run_workspace_v4_meter.py)

</details>

## Evidence

- [BOARD_RESULT_328277.json#L1](../../../submissions/328277/BOARD_RESULT_328277.json#L1)
- [receipt.json#L1](../twice-floor-variance-repair-legal-meter-2-126-977/receipt.json#L1)
- [METER_RESULT.json#L1](../../../submissions/328326/METER_RESULT.json#L1)
- [BOARD_RESULT_328326.json#L1](../../../submissions/328326/BOARD_RESULT_328326.json#L1)
- [SUBMIT_WATCH.log#L1](../../../submissions/328439/SUBMIT_WATCH.log#L1)
- [SUBMIT_WATCH.log#L1](SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (6)</summary>

- [BOARD_RESULT_328277.json](../../../submissions/328277/BOARD_RESULT_328277.json) · result
- [receipt.json](../twice-floor-variance-repair-legal-meter-2-126-977/receipt.json) · result
- [METER_RESULT.json](../../../submissions/328326/METER_RESULT.json) · result
- [BOARD_RESULT_328326.json](../../../submissions/328326/BOARD_RESULT_328326.json) · result
- [SUBMIT_WATCH.log](../../../submissions/328439/SUBMIT_WATCH.log) · log
- [SUBMIT_WATCH.log](SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-deployed-estimator-score-economics-minimum-charged-utilisation-verified) · [Copy provenance](../../../PROVENANCE.json)

