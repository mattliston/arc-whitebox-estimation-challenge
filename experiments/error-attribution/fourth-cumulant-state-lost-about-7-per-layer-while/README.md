# Fourth-cumulant state lost about 7% per layer while third-cumulant diagonals stayed accurate

2026-08-25 · experiment · informational · measured

verdict STATE, mechanism KAPPA4 PUSH LOSS: at 8M-sample Monte Carlo (MC) precision across three networks, the model's kappa3 diagonal is accurate and flat through layer 6 (median unit ratios 0.97-1.00) while kappa4 starts EXACT at layer 2 (0.999-1.016 - the emission is right) and decays about 7 percent per layer (0.93, 0.85-0.87, 0.78-0.80, 0.72-0.75 at layers 3-6), classification push_loss_decay in every network. Compounded to depth 15 the deep kappa4 runs at 40-50 percent of truth. Substituting oracle cumulants at layers 2-6 improves the mean at every layer.

Recorded measurement:

    SOLVED - verdict STATE, mechanism KAPPA4 PUSH LOSS: at 8M-sample MC precision across three networks, the model's kappa3 diagonal is accurate and flat through layer 6 (median unit ratios 0.97-1.00) while kappa4 starts EXACT at layer 2 (0.999-1.016 - the emission is right) and decays about 7 percent per layer (0.93, 0.85-0.87, 0.78-0.80, 0.72-0.75 at layers 3-6), classification push_loss_decay in every network. Compounded to depth 15 the deep kappa4 runs at 40-50 percent of truth. Substituting oracle cumulants at layers 2-6 improves the mean at every layer. The raw-MSE lane's named target: find and carry the kappa4 channel the push loses (successor A6A adjudicates CP4 dropped-plane mass versus un-carried generic-kappa4 content by measuring the same ratios on a cap-8192 variant and the carrier estimator).

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

- [ORACLE_KAPPA_RESULT.json#L1](ORACLE_KAPPA_RESULT.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [ORACLE_KAPPA_RESULT.json](ORACLE_KAPPA_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-fourth-cumulant-state-lost-about-7-per-layer-while) · [Copy provenance](../../../PROVENANCE.json)

