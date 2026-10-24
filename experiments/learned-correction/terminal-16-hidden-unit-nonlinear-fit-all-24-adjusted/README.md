# Terminal 16-hidden-unit nonlinear fit: all-24 adjusted -0.036487% but out-of-sample +0.005041%

2026-10-15T15:23:11Z · experiment · inconclusive · measured

A fixed small multilayer perceptron fit on 32 native training networks failed the unchanged out-of-sample sign rule despite a negative pooled result. The owner retained the exploratory source for possible stacking without standalone release qualification.

Recorded measurement:

    blocked CV -.169596% at shrink .5; full24 total -.105687% vs e406, increment -.036487% vs QR; OOS8 +.005041% vs QR

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_native_terminal_mlp16_v1/grade_retired_phase_v1.py](code/astra_native_terminal_mlp16_v1/grade_retired_phase_v1.py)
- [code/astra_native_terminal_mlp16_v1/build_native_export_v1.py](code/astra_native_terminal_mlp16_v1/build_native_export_v1.py)
- [code/astra_native_terminal_mlp16_v1/candidate_QR_terminal_mlp_ON_HELD_v1.py](code/astra_native_terminal_mlp16_v1/candidate_QR_terminal_mlp_ON_HELD_v1.py)
- [code/astra_native_terminal_mlp16_v1/fragment_terminal_mlp_HELD_v1.py](code/astra_native_terminal_mlp16_v1/fragment_terminal_mlp_HELD_v1.py)
- [code/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](code/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3402](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3402)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3400](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3400)

<details>
<summary>All result files and records (9)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [ALL24_DECISION_astra_v1.json](../linear-94-feature-ridge-nonlinear-94-16-1-fits/ALL24_DECISION_astra_v1.json) · result
- [BLOCKED_CV_RESULT_v1.json](../linear-94-feature-ridge-nonlinear-94-16-1-fits/BLOCKED_CV_RESULT_v1.json) · result
- [PILOT_DECISION_astra_v1.json](PILOT_DECISION_astra_v1.json) · result
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · result
- [ACTUAL_FIT_RETIREMENT_v1.json](ACTUAL_FIT_RETIREMENT_v1.json) · result
- [ALL32_FIT_ACTUAL_RETIREMENT_v1.json](ALL32_FIT_ACTUAL_RETIREMENT_v1.json) · result
- [ALL32_FIT_GUARD_v1.json](ALL32_FIT_GUARD_v1.json) · result
- [ALL32_FIT_READY_v1.json](ALL32_FIT_READY_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-terminal-16-hidden-unit-nonlinear-fit-all-24-adjusted) · [Copy provenance](../../../PROVENANCE.json)

