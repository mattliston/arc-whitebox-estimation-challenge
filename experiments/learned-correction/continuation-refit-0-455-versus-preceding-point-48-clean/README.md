# Continuation refit: -0.455% versus the preceding point on 48 clean ranks; a second step is rejected

2026-10-10T09:06Z · experiment · adopted · measured

jmD restarted the joint refit from candidate generation 44 (S44)'s PyTorch point (joint learned-array, head-input and covariance-fold refit (jmC) without local quartic of old pairs (qold)) with sizing and acceptance on a fresh 16 networks never in a gradient batch, mean-correction covariance-fold coefficient (kappa_C) as two arms (layers 1-9 / 10-14) and at most 4 steps; step 1 was accepted (-0.659% on the sizing set, the learned arrays the live arm) and step 2 stopped early (-0.337%, se 0.193), so jmD = S44 + one step, judged once on the 48 clean judging ranks at -0.455% against S44's PyTorch point and -2.513% against candidate generation 42 (S42). Its native build (sha 19251d3a) held parity on 651 / 378 and it became the first increment of candidate generation 45 (S45) (kappa_C 0.622 at layers 1-9, 0.924 at 10-14).

Recorded measurement:

    step 1 sizing 16: 1.3342e-08 vs 1.3430e-08 = -0.659% (se 0.195, 13/16); step 2 -0.337% (se 0.193, 9/16) not accepted; 48 clean vs S44 -0.455% (se 0.131, 35/48), vs S42 -2.513% (se 0.260, 44/48); all 50 vs S44 -0.469%; parity cosine 0.94-0.98 at layers 8-15, size ratio 1.01-1.04 at 10-15

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix10.sh](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix10.sh)
- [code/opus_reexam_20261001_v1/trainer/candidates/jmD/candidate.py](code/opus_reexam_20261001_v1/trainer/candidates/jmD/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/trainer/cert/fold/pairs.py](code/opus_reexam_20261001_v1/trainer/cert/fold/pairs.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials.py](code/opus_reexam_20261001_v1/trainer/export_dials.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L584](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L584)
- [TEAM_LOG_2026-10-01_to_10-22.md#L586](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L586)
- [TEAM_LOG_2026-10-01_to_10-22.md#L589](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L589)
- [TEAM_LOG_2026-10-01_to_10-22.md#L590](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L590)
- [TEAM_LOG_2026-10-01_to_10-22.md#L591](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L591)
- [TEAM_LOG_2026-10-01_to_10-22.md#L592](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L592)
- [TEAM_LOG_2026-10-01_to_10-22.md#L593](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L593)
- [TEAM_LOG_2026-10-01_to_10-22.md#L594](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L594)

<details>
<summary>All result files and records (7)</summary>

- [sz16_ranks.txt](sz16_ranks.txt) · result
- [jmD_p1_hu.json](jmD_p1_hu.json) · result
- [head_weights_31x128.json](head_weights_31x128.json) · result
- [physics_dials.json](physics_dials.json) · result
- [a_0mDpar_r0651.json](a_0mDpar_r0651.json) · result
- [runM_val48_clean.txt](runM_val48_clean.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-continuation-refit-0-455-versus-preceding-point-48-clean) · [Copy provenance](../../../PROVENANCE.json)

