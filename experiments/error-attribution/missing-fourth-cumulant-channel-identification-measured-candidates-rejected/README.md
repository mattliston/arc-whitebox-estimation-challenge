# Missing fourth-cumulant channel identification: measured candidates rejected

2026-08-25 · experiment · rejected · measured

Closed at the exact variant-identification spelling. ORACLE_KAPPA_VARIANTS.json (SHA-256 48cab175c5f06ac3024c070e7cf116455666f86da9c18deafbdb28b9fe7b4d73) returns MIXED under the frozen rule: cap-8192 pooled layer-6 recovery is -0.0631690786024732 (worse than base), while the rank-4 k22-residual carrier recovers only 0.1785429390176465. Neither reaches the 0.5 classification gate, and the three scalar calibration families also fail.

Recorded measurement:

    Closed at the exact variant-identification spelling. ORACLE_KAPPA_VARIANTS.json (SHA-256 48cab175c5f06ac3024c070e7cf116455666f86da9c18deafbdb28b9fe7b4d73) returns MIXED under the frozen rule: cap-8192 pooled layer-6 recovery is -0.0631690786024732 (worse than base), while the rank-4 k22-residual carrier recovers only 0.1785429390176465. Neither reaches the 0.5 classification gate, and the three scalar calibration families also fail. This closes CP4 drops versus this generic rank-4 carrier as an identifier; reopening requires a materially new carrier or observable, not another scalar gain.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/run_state_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/run_state_audit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/run_state_audit_deep.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k211_supply_v1/run_state_audit_deep.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_k22_carrier_ab_probe_v1/estimator_k22r.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_deep.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_deep.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_final_sensitivity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_final_sensitivity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_second_order.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_second_order.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_variants.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/analyze_variants.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/estimator_cap8192.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/estimator_cap8192.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_final_k5.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_final_k5.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_k5.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_k5.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_oracle.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_oracle.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_oracle_final.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_oracle_final.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_pairs.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/mc_pairs.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/pairs_spec.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/pairs_spec.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_deep_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_deep_audit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_final_tap.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_final_tap.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_probe.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_second_order_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_second_order_audit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_variants.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1/run_variants.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

</details>

## Evidence

- [CALIBRATION_FAMILY_CLOSURE.json#L1](CALIBRATION_FAMILY_CLOSURE.json#L1)
- [ORACLE_KAPPA_VARIANTS.json#L128](ORACLE_KAPPA_VARIANTS.json#L128)

<details>
<summary>All result files and records (2)</summary>

- [CALIBRATION_FAMILY_CLOSURE.json](CALIBRATION_FAMILY_CLOSURE.json) · result
- [ORACLE_KAPPA_VARIANTS.json](ORACLE_KAPPA_VARIANTS.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-missing-fourth-cumulant-channel-identification-measured-candidates-rejected) · [Copy provenance](../../../PROVENANCE.json)

