# Row-gain, carried-weight and radial readouts rejected after line searches

2026-10-10T02:27Z · experiment · rejected · measured

The three readout families had real gradients at layers 13-14 but their line searches bent at once; the row gain had no signal at all; none entered candidate generation 43 (S43).

Recorded measurement:

    carried b: step 0.0004 -0.092% (se 0.124), 0.0012 +0.75%, 0.0033 +9.85%; radial 0.025 -0.332% (se 0.269), 0.2 +6.75%; row gain mean t^2 1.28, 0 of 104 weights beyond 3 se

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/trainer/cert/head/readout_table.py](code/opus_reexam_20261001_v1/trainer/cert/head/readout_table.py)
- [code/opus_reexam_20261001_v1/trainer/cert/head/solve_rg.py](code/opus_reexam_20261001_v1/trainer/cert/head/solve_rg.py)
- [code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py](code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L480](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L480)
- [TEAM_LOG_2026-10-01_to_10-22.md#L490](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L490)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-row-gain-carried-weight-radial-readouts-rejected-after-line) · [Copy provenance](../../../PROVENANCE.json)

