# Join rank cuts and grows at layers 7, 10, 13 (320, 256, 448): cuts lose more accuracy than bill, grows pay under 1:1

2026-10-16T19:03:49Z · experiment · rejected · measured

The rank of the joined plane basis was moved as a value-only edit of the estimator submitted as #336412 (S2tr): layer 7 to 320 cut the bill by about 1% but cost +1.69% MSE on 21 store networks; layer 13 to 320 and layer 7 to 256 failed on 8 seeds; growing layer 10 or 13 to 448 cost +2.25% / +0.73% bill for no store accuracy gain; layer 10 back to 320 cost +2.69% MSE with +887 warm requests as shapes fell off the prebind list. Join rank trades worse than 1:1 at every layer, consistent with the earlier rank law.

Recorded measurement:

    r7rank320 (600d065d): warm-judge bill -0.854% / -1.027% first/warm, requests +255 / -25, client instructions -6.4%; S8 xMSE +0.748% (SE .64, 3/8), bill -0.989%, xnet -0.242% (SE .61); store n21 xMSE +1.693% (SE .47, 7/21 lower), bill -1.07, xnet +0.62% (pooled +0.30). r13rank320: bill -0.345% / -0.495%, requests +442 / +226, S8 xMSE +1.5%. r7rank256: S8 xMSE +5.4%, bill -2.3%. r10rank448: bill +2.42%, requests +1,495/+1,272, store n5 xMSE +0.35 (SE .40); r13rank448: +0.78% / +0.61%, store +0.13 (SE .21). r10rank320: MSE +2.69% on 7103256, bill -0.9%, requests +887 warm.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_solo_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_r7rank320.py](code/cand_r7rank320.py)
- [code/cand_r10rank448.py](code/cand_r10rank448.py)
- [code/cand_r13rank448.py](code/cand_r13rank448.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4448](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4448)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4455](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4455)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4460](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4460)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4462](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4462)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4465](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4465)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4467](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4467)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4469](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4469)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4471](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4471)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4511](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4511)

<details>
<summary>All result files and records (5)</summary>

- [GRADE_r7rank320.txt](GRADE_r7rank320.txt) · result
- [GRADE_r13rank320.txt](GRADE_r13rank320.txt) · result
- [GRADE_r7rank256.txt](GRADE_r7rank256.txt) · result
- [fragment_r7rank320_v1.py.txt](fragment_r7rank320_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-join-rank-cuts-grows-layers-7-10-13-320) · [Copy provenance](../../../PROVENANCE.json)

