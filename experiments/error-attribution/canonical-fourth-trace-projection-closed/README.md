# Canonical fourth-trace projection: closed

2026-09-06T10:47Z · experiment · rejected · measured

2026-09-06 03:45:37 PDT, Phase 2. The canonical Euclidean harmonic projection was independently certified against explicit symmetric tensors: adjoint/trace/residual orthogonality errors<=2.67e-15; Pythagoras<=2.85e-14. No metric/source/scale/rank/layer/count/seed sweep, sampling expansion, carrier, fit, port, panel or submission.

Recorded measurement:

    2026-09-06 03:45:37 PDT, Phase 2. The canonical Euclidean harmonic projection was independently certified against explicit symmetric tensors: adjoint/trace/residual orthogonality errors<=2.67e-15; Pythagoras<=2.85e-14. This is genuinely the orthogonal trace-generated component, unlike the previous covariance-metric trace-matching reconstruction. No new samples. Same rank35 post12->pre13->mean response14, two existing independent8M statistic files, same actual bank frozen inputs. All six target response arrays are byte-identical to the prior oracle audit. Core reconstruction6.18e-17 relative, direct pair response3.22e-15 absolute. | Consumer | Residual/baseline signal cross-energy | Cross A->B | Cross B->A | |---|---:|---:|---:| | Covariance response | 0.6336143110 | 0.6454972814 | 0.6434175949 | | Next mean response | 0.6108077062 | 0.6269336997 | 0.6193154809 | Both fail required signal ratio<=0.50, despite ordinary cross ratios<=0.65. Signal/noise remains42.73/39.37. These are isolated fourth-contribution response comparisons, NOT total error or an adjusted/raw estimator score. The canonical projection's orthogonality does not imply downstream value optimality under these nonlinear, weight-dependent consumers.

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

- [CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md#L1](CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md#L1)
- [CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md#L5](CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md#L5)
- [CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md#L17](CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md#L17)

<details>
<summary>All result files and records (3)</summary>

- [CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md](CANONICAL_FOURTH_TRACE_VALUE_V1_CLOSED.md) · record
- [canonical_fourth_trace_value_v1.json](canonical_fourth_trace_value_v1.json) · result
- [CANONICAL_FOURTH_TRACE_VALUE_DESIGN_V1.md](CANONICAL_FOURTH_TRACE_VALUE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-canonical-fourth-trace-projection-closed) · [Copy provenance](../../../PROVENANCE.json)

