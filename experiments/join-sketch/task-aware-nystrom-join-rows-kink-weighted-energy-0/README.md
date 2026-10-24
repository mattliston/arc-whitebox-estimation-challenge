# Task-aware Nystrom join rows (kink-weighted energy): -0.455% on 8 seeds, +0.213% on 32 store networks, removed

2026-10-16T18:04:54Z · experiment · rejected · measured

Option B multiplies the energy of each Nystrom join row at the layer-10 and layer-13 joins by (1 + beta t_j / mean t), where t_j is the squared weight of Gram row j into the layer kink units (|a| < 1), before the top-rank row selection (on top of the energy order JORD). It passed the 8-seed screen at beta 1, read -0.208% on 24 seeds and -0.160% pooled over 47 seeds, was stacked into S4/S5 and the frontier builds, and was then removed when the 32 independent store networks read +0.213% against JORD (pooled about +0.12%): the 8-seed read was winner s curse, and its change of the kink variance runs along the true error with share only 0.3%. Variants at joins 7/10/13 and with a soft kink weight were no better.

Recorded measurement:

    S8 (paired vs screen_jord): beta 1 (706cd723) xnet -0.455% (se .29), 6/8 lower, bill +0.007%; beta 2 -0.311% (se .27). 24 seeds: B vs JORD -0.208% (se .198), 14/24 lower (S8 -0.455, A80 -0.084). 47 seeds pooled: -0.160% (se .132), xMSE -0.167%, 29/47 lower. Store: 8 ranks B vs JORD +0.256% (SE .14, 6/8 worse); 32 ranks +0.213% (SE .113, 12/32 lower), vs S2tr +0.252% (SE .22); pooled ~+0.12% vs JORD (SE .10). bj7 (joins 7/10/13) vs B +0.015% (SE .126); bsoft vs B +0.157% (SE .196).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2h_v1/build_tjoin_v1.py](code/opus_ladder2h_v1/build_tjoin_v1.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_tj10b1_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_tj10b1_v1.py)
- [code/opus_passrel_v1/tank_v1/build_tank_v1.py](code/opus_passrel_v1/tank_v1/build_tank_v1.py)
- [code/opus_passrel_v1/tank_v1/cand_s2tr_jord_tank_b10_v1.py](code/opus_passrel_v1/tank_v1/cand_s2tr_jord_tank_b10_v1.py)
- [code/opus_passrel_v1/tank_v1/pool48_v1.py](code/opus_passrel_v1/tank_v1/pool48_v1.py)
- [code/opus_kinksel_v1/cand_tank_bj7_v1.py](code/opus_kinksel_v1/cand_tank_bj7_v1.py)
- [code/opus_kinksel_v1/cand_tank_bsoft_v1.py](code/opus_kinksel_v1/cand_tank_bsoft_v1.py)
- [code/opus_passrel_v1/tank_v1/cand_s2tr_jord_v1.py](code/opus_passrel_v1/tank_v1/cand_s2tr_jord_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4336](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4336)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4343](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4343)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4349](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4349)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4360](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4360)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4371](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4371)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4374](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4374)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4404](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4404)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4405](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4405)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4416](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4416)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4418](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4418)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4434](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4434)

<details>
<summary>All result files and records (2)</summary>

- [frag_nob_v1.py.txt](frag_nob_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- rejects: [Store-32 validation: every value-only part picked on generated seeds reads about zero or worse on 32 store networks](../../../records/research/2026-10-16-store-32-validation-every-value-only-part-picked-generated/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-task-aware-nystrom-join-rows-kink-weighted-energy-0) · [Copy provenance](../../../PROVENANCE.json)

