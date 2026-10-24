# Banded continuation refit rejected: -0.313% sizing MSE but -0.121% leave-one-network-out

2026-10-10T09:27Z · experiment · rejected · measured

The continuation refit labelled jmE attempted learned-array and early/late covariance-fold arms from the prior accepted point. Its first step failed the predeclared early-stop rule and the prior point remained unchanged.

Recorded measurement:

    16 ranks: 1.3613e-08 against 1.3655e-08: -0.313% (se 0.218, better on 8/16; all layers -0.188%); model at its optimum -0.327%, leave-one-network-out -0.121%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/trainer/cert/fold/pairs.py](code/opus_reexam_20261001_v1/trainer/cert/fold/pairs.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix11.sh](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix11.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_v7_before_bands.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_v7_before_bands.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials.py](code/opus_reexam_20261001_v1/trainer/export_dials.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L594](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L594)
- [TEAM_LOG_2026-10-01_to_10-22.md#L596](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L596)
- [TEAM_LOG_2026-10-01_to_10-22.md#L597](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L597)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [sz16b_ranks.txt](sz16b_ranks.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-banded-continuation-refit-rejected-0-313-sizing-mse-but) · [Copy provenance](../../../PROVENANCE.json)

