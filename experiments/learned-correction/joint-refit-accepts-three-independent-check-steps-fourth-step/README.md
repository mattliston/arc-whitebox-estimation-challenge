# Joint refit accepts three independent-check steps; fourth step rejected at +0.016% MSE

2026-10-10T12:29Z · experiment · adopted · measured

The joint continuation refit labelled jmG fitted learned-array, head-input and covariance-fold arms on one set and checked each point on an independent set. Three steps were accepted; the fourth increased independent-check error and was discarded before the final held-out assessment.

Recorded measurement:

    Independent 16: step 1 -0.408% (se 0.389, better on 9/16); step 2 -0.352% (se 0.127, better on 11/16); step 3 -0.312% (se 0.134, better on 12/16); step 4 +0.016% (se 0.101, better on 7/16)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/trainer/cert/fold/pairs.py](code/opus_reexam_20261001_v1/trainer/cert/fold/pairs.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix13.sh](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix13.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_v9_before_dial_arms.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_v9_before_dial_arms.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials.py](code/opus_reexam_20261001_v1/trainer/export_dials.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L613](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L613)
- [TEAM_LOG_2026-10-01_to_10-22.md#L614](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L614)
- [TEAM_LOG_2026-10-01_to_10-22.md#L620](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L620)
- [TEAM_LOG_2026-10-01_to_10-22.md#L625](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L625)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [jmG_p1_hu.json](jmG_p1_hu.json) · result
- [jmG_p2_hu.json](jmG_p2_hu.json) · result
- [jmG_p3_hu.json](jmG_p3_hu.json) · result
- [sz16d_ranks.txt](sz16d_ranks.txt) · result
- [sz16e_ranks.txt](sz16e_ranks.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-joint-refit-accepts-three-independent-check-steps-fourth-step) · [Copy provenance](../../../PROVENANCE.json)

