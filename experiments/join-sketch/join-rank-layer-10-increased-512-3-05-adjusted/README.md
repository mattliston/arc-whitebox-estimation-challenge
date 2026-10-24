# Join rank at layer 10 increased to 512: +3.05% adjusted for -0.93% MSE and +4.00% FLOPs

2026-10-20T03:15:49Z · experiment · rejected · measured

The 32-network test rejected increased join rank at layer 10 because the accuracy gain did not pay for its compute cost.

Recorded measurement:

    - [fable-lead] 2026-10-20T03:15:49Z JOIN RANK L10 512 (r10_512, 32 store ranks vs stk1_r79): raw -0.93% for bill +4.00% = adjusted +3.05% (0/28 lower); the join-rank law holds (1:1 or worse); r1013_512 / r10_448 / r71013_512 dropped from the queue, r7_512 (running) kept as the last check. #337247 VALID 1.428880e-9 is the new best.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_r10_512_sci_v1.py](code/candidate_r10_512_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6125](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6125)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-join-rank-layer-10-increased-512-3-05-adjusted) · [Copy provenance](../../../PROVENANCE.json)

