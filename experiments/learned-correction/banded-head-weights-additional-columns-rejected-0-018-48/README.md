# Banded head weights and additional columns rejected: -0.018% on 48 clean ranks; quartic fold costs +1.343%

2026-10-10T12:53Z · experiment · rejected · measured

Separate column-6 / column-20 weights at layers 10-14 sit at their optimum (0 of 128 beyond 2.5 se); the five cheap adjoint-screen columns are worth -0.018% on the 48 clean; fold (4th-chaos diagonal of pairs aged 3+) adds -0.36% for +1.34% metered FLOPs.

Recorded measurement:

    jmE late rows mean t^2 0.70 / 0.50; batch-7 cheap five -0.018% (se 0.020, 28 of 48) at +143 requests; fold -0.358% (se 0.196) for +1.343% bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/t64_adjoint_screen.py](code/opus_reexam_20261001_v1/physicist/t64_adjoint_screen.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py](code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py](code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L594](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L594)
- [TEAM_LOG_2026-10-01_to_10-22.md#L598](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L598)
- [TEAM_LOG_2026-10-01_to_10-22.md#L602](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L602)
- [TEAM_LOG_2026-10-01_to_10-22.md#L604](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L604)
- [TEAM_LOG_2026-10-01_to_10-22.md#L605](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L605)
- [TEAM_LOG_2026-10-01_to_10-22.md#L626](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L626)
- [TEAM_LOG_2026-10-01_to_10-22.md#L627](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L627)
- [TEAM_LOG_2026-10-01_to_10-22.md#L630](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L630)

<details>
<summary>All result files and records (2)</summary>

- [t64_screen_36_full.txt](../../../instruments/adjoint-screen-computes-exact-zero-weight-gradients-candidate-head/t64_screen_36_full.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-banded-head-weights-additional-columns-rejected-0-018-48) · [Copy provenance](../../../PROVENANCE.json)

