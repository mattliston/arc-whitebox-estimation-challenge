# Covariance-edge fourth-order core: both declared truncation gates failed

2026-09-05T22:32Z · experiment · rejected · measured

Cost v1 omitted expansion of the bank's active input repeated plane to the full support required by the new query. V2 corrects that. - Nonradial R@U relative truncation error .651031014815 (gate<=.15 FAIL). - L12 core -> L13 source -> actual W14 read relative truncation error .235246985248 (gate<=.15 FAIL). - Polynomial core query residual4.914755935e-8 (gate1e-6 PASS). - Mixed query/core differences3.102800656e-5/3.269723575e-5 (gate.01 PASS on this cell). - Residual covariance Frobenius8.45453, min inverse shift100.38372, Chebyshev ratio .0848282, system condition1499.78. - Full incremental trace-query norm2.43661, anisotropic core norm.135318, source next-diagonal norm.000346357. - Whole diagnostic6.071s, cell1.574s, peakRSS4,765,351,936bytes, one nice10 CPU worker with three BLAS threads.

Recorded measurement:

    Cost v1 omitted expansion of the bank's active input repeated plane to the full support required by the new query. V2 corrects that. V3 compiles the non-third repeated-correction polynomial exactly (primary certificate: 66 canonical monomials in13 matrix groups). V3 projected multipliers: all-f64 1.53010/1.53086 (FAIL); mixed 1.39293/1.39342 (PASS original1.40 gate), ranks35/100. These are analytical architecture projections, not actual bills. Audit note: its generic non-CP star contraction is priced on previous active support; cold diagonal-star contractions would also need a separately accounted temporary CP branch or a full-support read. This unresolved small addition cannot be silently ignored in deployment. No release inference is based on the cost gate alone. - Nonradial R@U relative truncation error .651031014815 (gate<=.15 FAIL). - L12 core -> L13 source -> actual W14 read relative truncation error .235246985248 (gate<=.15 FAIL). - Polynomial core query residual4.914755935e-8 (gate1e-6 PASS). - Mixed query/core differences3.102800656e-5/3.269723575e-5 (gate.01 PASS on this cell). - Residual covariance Frobenius8.45453, min inverse shift100.38372, Chebyshev ratio .0848282, system condition1499.78. - Full incremental trace-query norm2.43661, anisotropic core norm.135318, source next-diagonal norm.000346357. - Whole diagnostic6.071s, cell1.574s, peakRSS4,765,351,936bytes, one nice10 CPU worker with three BLAS threads. No remaining worker.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_thin_fourth_wide_v1.py](code/astra_phase2_reexamination_v1/check_thin_fourth_wide_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_repeated_compiled_v1.py](code/astra_phase2_reexamination_v1/fourth_repeated_compiled_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_trace_core_projection_v1.py](code/astra_phase2_reexamination_v1/fourth_trace_core_projection_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_trace_full_edge_reference_v1.py](code/astra_phase2_reexamination_v1/fourth_trace_full_edge_reference_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_trace_full_queries_v1.py](code/astra_phase2_reexamination_v1/fourth_trace_full_queries_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_trace_fused_cp_queries_v1.py](code/astra_phase2_reexamination_v1/fourth_trace_fused_cp_queries_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_trace_third_queries_v1.py](code/astra_phase2_reexamination_v1/fourth_trace_third_queries_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/thin_fourth_wide_helpers_v1.py](code/astra_phase2_reexamination_v1/thin_fourth_wide_helpers_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [THIN_FOURTH_Q4S4_V1_CLOSED.md#L1](THIN_FOURTH_Q4S4_V1_CLOSED.md#L1)
- [THIN_FOURTH_Q4S4_V1_CLOSED.md#L15](THIN_FOURTH_Q4S4_V1_CLOSED.md#L15)
- [THIN_FOURTH_Q4S4_V1_CLOSED.md#L34](THIN_FOURTH_Q4S4_V1_CLOSED.md#L34)

<details>
<summary>All result files and records (2)</summary>

- [THIN_FOURTH_Q4S4_V1_CLOSED.md](THIN_FOURTH_Q4S4_V1_CLOSED.md) · record
- [thin_fourth_wide_r35_v1.json](thin_fourth_wide_r35_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-covariance-edge-fourth-order-core-both-declared-truncation-gates) · [Copy provenance](../../../PROVENANCE.json)

