# 64-query terminal-summary proxy fails drift bounds: RMS 1.5409484511173072e-5 versus 5e-7

2026-09-07 · experiment · rejected · measured

A fixed query-summary control variate approximated four terminal response features while retaining exact linear features. It failed a sufficient-small-perturbation guard; no target-error measurement established an adjusted-score regression.

Recorded measurement:

    RMS drift1.5409484511173072e-5 versus5e-7;maxabs9.08970832824707e-5 versus5e-6;planning cost463461267267F versus472173398147F,not a measured replacement bill.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/probe_terminal_summary64_v3.py](code/probe_terminal_summary64_v3.py)
- [code/capture_terminal_summary_v2.py](code/capture_terminal_summary_v2.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_pair_polynomial_f32_v1.py](code/candidate_response_pair_polynomial_f32_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/capture_terminal_summary_v1.py](code/capture_terminal_summary_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/probe_terminal_summary64_v1.py](code/probe_terminal_summary64_v1.py)
- [code/probe_terminal_summary64_v2.py](code/probe_terminal_summary64_v2.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)

</details>

## Evidence

- [TERMINAL_SUMMARY64_FIXED_PROXY_NOT_ADVANCED_V1.md#L15](TERMINAL_SUMMARY64_FIXED_PROXY_NOT_ADVANCED_V1.md#L15)

<details>
<summary>All result files and records (2)</summary>

- [TERMINAL_SUMMARY64_FIXED_PROXY_NOT_ADVANCED_V1.md](TERMINAL_SUMMARY64_FIXED_PROXY_NOT_ADVANCED_V1.md) · record
- [TERMINAL_SUMMARY64_FUNCTIONAL_REPAIR_V3.md](TERMINAL_SUMMARY64_FUNCTIONAL_REPAIR_V3.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-64-query-terminal-summary-proxy-fails-drift-bounds-rms) · [Copy provenance](../../../PROVENANCE.json)

