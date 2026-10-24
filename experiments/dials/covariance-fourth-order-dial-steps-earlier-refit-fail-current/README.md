# Covariance and fourth-order dial steps from an earlier refit fail on the current submitted source

2026-10-16T16:01:53Z · experiment · rejected · measured

The three dial moves pre-registered for the S63 refit (never evaluated) were applied as value-only overrides on the estimator submitted as #336412 (S2tr), each the native value plus the 0.02 step in the descent direction: the covariance-channel kc at layers 7-10, the fourth-order birth coefficient x2a1 at layer 7, and the ages-3 fourth-order strength at layers 4-7. All failed the -0.15% bar. The gradients were taken at S60 and S2tr sits at a later point (kc L7 0.477 vs 0.422, x2a1 L7 0.473 vs 0.262, k4la3 L4-7 off vs -1.04), so the moves do not transfer.

Recorded measurement:

    KC (e5df5d69): xnet +0.010% (SE .083, 3/8 lower); X2A1 (695615f0): +0.100% (SE .064, 3/8); K4LA3 (294eea0b): n3 xnet +0.498% (bill +0.04%, 0/3).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_passrel_v1/s63dial_v1/build_s63dial_v1.py](code/opus_passrel_v1/s63dial_v1/build_s63dial_v1.py)
- [code/opus_passrel_v1/s63dial_v1/cand_s2tr_s63_KC_v1.py](code/opus_passrel_v1/s63dial_v1/cand_s2tr_s63_KC_v1.py)
- [code/opus_passrel_v1/s63dial_v1/cand_s2tr_s63_X2A1_v1.py](code/opus_passrel_v1/s63dial_v1/cand_s2tr_s63_X2A1_v1.py)
- [code/opus_passrel_v1/s63dial_v1/cand_s2tr_s63_K4LA3_v1.py](code/opus_passrel_v1/s63dial_v1/cand_s2tr_s63_K4LA3_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4308](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4308)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4314](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4314)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4317](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4317)

<details>
<summary>All result files and records (2)</summary>

- [LADDER2_GRADE_latest.json](LADDER2_GRADE_latest.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-covariance-fourth-order-dial-steps-earlier-refit-fail-current) · [Copy provenance](../../../PROVENANCE.json)

