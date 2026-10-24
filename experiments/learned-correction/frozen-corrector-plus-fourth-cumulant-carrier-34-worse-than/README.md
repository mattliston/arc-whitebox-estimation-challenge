# Frozen corrector plus fourth-cumulant carrier: 34% worse than the deployed baseline

2026-08-25 · experiment · rejected · measured

The combined card (banked base + t7 trim + rank-4 carrier + output clamp, cholesky-free, three audit rounds) scored 3.974e-8 deployed against the raw host (minus 56 percent, 200 of 200) but +34 percent AGAINST THE TRUE DEPLOYED BASELINE (frozen corrector on the sealed base tapes: 2.958e-8 on the same rows, matching its 2.941e-8 board grade): the frozen learned corrector and the analytic carrier double-remove their shared component. Closed without firing; the licensing baseline mis-specification is on record. Deployment law hardened: fires are judged only against the deployed banked card on the same rows.

Recorded measurement:

    The combined card (banked base + t7 trim + rank-4 carrier + output clamp, cholesky-free, three audit rounds) scored 3.974e-8 deployed against the raw host (minus 56 percent, 200 of 200) but +34 percent AGAINST THE TRUE DEPLOYED BASELINE (frozen corrector on the sealed base tapes: 2.958e-8 on the same rows, matching its 2.941e-8 board grade): the frozen learned corrector and the analytic carrier double-remove their shared component. Closed without firing; the licensing baseline mis-specification is on record. Deployment law hardened: fires are judged only against the deployed banked card on the same rows.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_card_v1/build_estimator_combined.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_card_v1/build_estimator_combined.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_pilot_v1/generate_row.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_pilot_v1/generate_row.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_pilot_v1/run_generation.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_pilot_v1/run_generation.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_pilot_v1/score_pilot.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_combined_pilot_v1/score_pilot.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/build_estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [PILOT_RAW_RESULT.json#L1](PILOT_RAW_RESULT.json#L1)
- [COMPOSITION_CLOSURE.json#L3](COMPOSITION_CLOSURE.json#L3)
- [AUDIT_RECEIPT_v3.json#L1](AUDIT_RECEIPT_v3.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [PILOT_RAW_RESULT.json](PILOT_RAW_RESULT.json) · result
- [COMPOSITION_CLOSURE.json](COMPOSITION_CLOSURE.json) · result
- [AUDIT_RECEIPT_v3.json](AUDIT_RECEIPT_v3.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-frozen-corrector-plus-fourth-cumulant-carrier-34-worse-than) · [Copy provenance](../../../PROVENANCE.json)

