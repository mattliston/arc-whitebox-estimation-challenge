# Join-sketch columns reduced to 1536/1024/1024: -0.390% adjusted; smaller counts lose raw accuracy

2026-10-19T07:43:09Z · experiment · adopted · measured

The number of sketched columns (kc) of the Nystrom join sketch (JSK) at the three joins (layers 7, 10, 13) was cut from 2048/1536/1536 to 1536/1024/1024 with factors and read kept exact: on 32 paired ranks the bill falls 0.297% on every rank and raw improves 0.094% (held-out 25: -0.482%), a step; 2048/1024/1024 is half as good, and 1024/1024/1024 or 1536/768/768 start to cost raw, so the line search closed at the fired setting. A two-rank read on 10-18 had called the same squeeze +0.04%.

Recorded measurement:

    kc 1536/1024/1024 (f45d12bb) vs the base a7ffd713: adjusted -0.390% (se .111, 25/32 lower) = bill -0.297% (every rank) + raw -0.094%; held-out 25 adjusted -0.482%, raw -0.185%; kc 2048/1024/1024 (76a1798c): adjusted -0.183% (bill -0.220%, raw +0.038%); the seven ranks read raw +0.23% for both; 1024/1024/1024 (083af95c) vs base -0.285% (bill -0.395%, raw +0.112%) = +0.106% vs the fired file (raw +0.206%, bill -0.099%); 1536/768/768 (c785f20f, 25 ranks): +0.102% vs fired (se .15; raw +0.175%, bill -0.072%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_kc_1536_1024_1024_sci_v1.py](code/candidate_kc_1536_1024_1024_sci_v1.py)
- [code/candidate_kc_2048_1024_1024_sci_v1.py](code/candidate_kc_2048_1024_1024_sci_v1.py)
- [code/candidate_kc_1024_1024_1024_sci_v1.py](code/candidate_kc_1024_1024_1024_sci_v1.py)
- [code/candidate_kc_1536_768_768_sci_v1.py](code/candidate_kc_1536_768_768_sci_v1.py)
- [code/candidate_rel_ab47_kc1536_1024_1024_v1.py](code/candidate_rel_ab47_kc1536_1024_1024_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5919](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5919)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5920](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5920)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5923](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5923)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5924](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5924)

<details>
<summary>All result files and records (2)</summary>

- [RESULT32_v1.json](RESULT32_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337065: join sketch columns 1536/1024/1024 on #337061, graded 1.4390249e-9 (-0.252%, full bill transfer)](../../../submissions/337065/README.md)
- ← rejects: [Submission #337028 audit finds about 0.4% FLOP savings and no measured change covering the 1.7% score gap](../../../records/research/2026-10-19-submission-337028-audit-finds-about-0-4-flop-savings/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-join-sketch-columns-reduced-1536-1024-1024-0-390) · [Copy provenance](../../../PROVENANCE.json)

