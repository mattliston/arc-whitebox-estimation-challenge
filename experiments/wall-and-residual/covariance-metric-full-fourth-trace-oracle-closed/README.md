# Covariance-metric full fourth-trace oracle: closed

2026-09-06T10:45Z · experiment · rejected · measured

2026-09-06 03:40:50 PDT, Phase 2. Both fixed existing independent8M streams passed new-statistic qualification. The canonical Euclidean harmonic reconstruction Sym(A,I), explicitly specified in the earlier full-carrier proposal, was NOT tested here.

Recorded measurement:

    2026-09-06 03:40:50 PDT, Phase 2. Both fixed existing independent8M streams passed new-statistic qualification. Post12/pre13 shifted means reproduced exactly. R-diagonal independent reduction differences were 0.0482%/0.0481%; pre13 K22/K31 marginal-diagonal checks about0.192%. Acquisition72.881s/76.679s, one MPS job each, peak hostRSS1.263/1.283GB. Shared float32 numerical bias remains possible; replicas remove only independent sampling noise from cross-energy estimates. The actual bank inputs were frozen first, prefix13 comparisons byte-exact, independent pair baseline max6.94e-18. Oracle core reconstruction max 4.49e-15 relative; finite response vs direct pair max4.47e-15 absolute. | Consumer | Residual/baseline replica cross-energy | Cross A->B | Cross B->A | Signal/noise | |---|---:|---:|---:|---:| | Covariance response | 0.6135335245 | 0.6257960528 | 0.6231072761 | 42.73 | | Next mean response | 0.6112197142 | 0.6287411502 | 0.6175581810 | 39.37 | Required cross-energy ratios <=0.50 and both ordinary cross ratios<=0.65. Both fail the first requirement despite qualifying signal and the ordinary ratio requirements. The ~39% isolated response-error reduction is not an adjusted or raw score improvement. No fit, larger sample/net/layer study, carrier, port, panel, or submission under this design. Receipt fourth_trace_oracle_value_v1.json/.npz. The two new statistic files are fourth_trace_oracle_r35_A_v1.npz and B, with hashes recorded in the receipt. Bank#329926 is unchanged at4.982231460183304e-9 adjusted.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/canonical_fourth_trace_v1.py](code/astra_phase2_reexamination_v1/canonical_fourth_trace_v1.py)
- [code/astra_phase2_reexamination_v1/check_canonical_fourth_trace_value_v1.py](code/astra_phase2_reexamination_v1/check_canonical_fourth_trace_value_v1.py)
- [code/astra_phase2_reexamination_v1/check_fourth_trace_oracle_value_v1.py](code/astra_phase2_reexamination_v1/check_fourth_trace_oracle_value_v1.py)
- [code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py](code/astra_phase2_reexamination_v1/current_pair_oracle_math_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_trace_distinct_necessity_v1.py](code/astra_phase2_reexamination_v1/fourth_trace_distinct_necessity_v1.py)
- [code/astra_phase2_reexamination_v1/fourth_trace_distinct_necessity_v2.py](code/astra_phase2_reexamination_v1/fourth_trace_distinct_necessity_v2.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py](code/astra_phase2_reexamination_v1/quadratic_skew_covariance_v1.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/thin_fourth_wide_helpers_v1.py](code/astra_phase2_reexamination_v1/thin_fourth_wide_helpers_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L1](FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L1)
- [FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L7](FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L7)
- [FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L13](FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L13)
- [FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L24](FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md#L24)

<details>
<summary>All result files and records (3)</summary>

- [FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md](FOURTH_TRACE_ORACLE_VALUE_V1_CLOSED.md) · record
- [fourth_trace_oracle_value_v1.json](fourth_trace_oracle_value_v1.json) · result
- [FOURTH_TRACE_ORACLE_VALUE_DESIGN_V1.md](FOURTH_TRACE_ORACLE_VALUE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-covariance-metric-full-fourth-trace-oracle-closed) · [Copy provenance](../../../PROVENANCE.json)

