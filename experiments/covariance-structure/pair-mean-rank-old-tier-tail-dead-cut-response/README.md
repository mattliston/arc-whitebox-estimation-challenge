# Pair-mean rank, old-tier tail, dead-cut and response-floor cuts all lost more accuracy than bill

2026-10-17T20:21:10Z · experiment · rejected · measured

Value-only reductions of pair-mean frame rank, old-tier reconstruction threshold, dead-unit threshold and response-roster floor were screened against the repaired estimator. The first batch had contaminated bills but decisive raw-error increases; the clean second batch confirmed unfavorable accuracy-to-cost tradeoffs.

Recorded measurement:

    Pair-mean rank 128: raw +21.9%; rank 192: +4.0%; tail z 2.5→2.0: raw +14.3%; dead cut 3.09→2.5: raw +2.6%; pair truncation at layer 7: bill -0.40%, raw +1.18%; response floor .02: bill -1.26%, raw +5.6%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cheap/cand_dc25.py](code/cheap/cand_dc25.py)
- [code/cheap/cand_lcr128.py](code/cheap/cand_lcr128.py)
- [code/cheap/cand_lcr192.py](code/cheap/cand_lcr192.py)
- [code/cheap/cand_lcr7.py](code/cheap/cand_lcr7.py)
- [code/cheap/cand_ott20.py](code/cheap/cand_ott20.py)
- [code/cheap/cand_rrf02.py](code/cheap/cand_rrf02.py)
- [code/grade_store_v1.py](code/grade_store_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4916](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4916)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4919](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4919)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [frag_lcr128.py.txt](frag_lcr128.py.txt) · code
- [frag_lcr192.py.txt](frag_lcr192.py.txt) · code
- [frag_ott20.py.txt](frag_ott20.py.txt) · code
- [frag_dc25.py.txt](frag_dc25.py.txt) · code
- [frag_lcr7.py.txt](frag_lcr7.py.txt) · code
- [frag_rrf02.py.txt](frag_rrf02.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-pair-mean-rank-old-tier-tail-dead-cut-response) · [Copy provenance](../../../PROVENANCE.json)

