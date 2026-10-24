# Safe request geometry: two-strip plus thin plain products costs 1.046% and leaves 498 warm-request margin

2026-10-16T02:19:05Z · experiment · inconclusive · measured

Pricing compared the canonical five-full-product safe construction with a thin-product plain-matmul alternative. The latter was cheaper and safer by the measured gates, but safe preparation was parked at the owner request.

Recorded measurement:

    two-striponlywarm43069,+.670%bill;canonical42619,+1.138%,81margin;thinplain42202,498margin,first44255,+1.046%,worst105.2s,risk3.23%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_safecost_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_S2t_TS_thin_v1.py](code/cand_S2t_TS_thin_v1.py)
- [code/cand_S2t_TS_v1.py](code/cand_S2t_TS_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3696](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3696)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3689](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3689)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3698](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3698)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_thin_plain_v1.py.txt](fragment_thin_plain_v1.py.txt) · result
- [SAFECOST_v1.json](SAFECOST_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-safe-request-geometry-two-strip-plus-thin-plain-products) · [Copy provenance](../../../PROVENANCE.json)

