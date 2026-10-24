# Gaussian early termination reduced bill but increased final error by 1,613% to 123,169%

2026-10-17T18:18:11Z · experiment · rejected · measured

A hard residual guard replaced the remaining moment corrections with a Gaussian tail at specified layers. Every forced tail lost enough final accuracy that one affected network would dominate the run mean; safety remained a matter of runtime margin and controlled collection.

Recorded measurement:

    Rank 35 raw U: layer 6 +123,169%, layer 10 +53,740%, layer 13 +11,710%, layer 14 +3,040%, layer 15 +1,613%; layer-6 requests 45,090→15,897, bill -66%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_guard_FORCED2_test.py](code/candidate_guard_FORCED2_test.py)
- [code/candidate_guard_v37h2.py](code/candidate_guard_v37h2.py)
- [code/grade_pair_v1.py](code/grade_pair_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4897](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4897)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [fragment_hard_guard_v2.py.txt](fragment_hard_guard_v2.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-gaussian-early-termination-reduced-bill-but-increased-final-error) · [Copy provenance](../../../PROVENANCE.json)

