# Rank-384 stack census refit: 977 fewer first and 351 fewer warm requests

2026-10-16T05:03:25Z · experiment · adopted · measured

The source-specific prebind list, accumulator refit and least-recently-used strip caches preserved all predictions and reduced lazy binding costs. The repaired rank 384 stack was submitted with remaining residual-tail risk.

Recorded measurement:

    130swaps;requests-977/-351normal,-915/-321slow;FLOPs-.36%first/-.08%warm;risk17.44to13.38%;worst109.5s;slow4.580s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_s2tr_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_S2tr_v1.py](code/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3891](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3891)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [S2TR_DECISION_v1.json](S2TR_DECISION_v1.json) · result
- [fragment_ks_acc_s2t_v1.py.txt](fragment_ks_acc_s2t_v1.py.txt) · result
- [fragment_prebind_s2t_v1.py.txt](fragment_prebind_s2t_v1.py.txt) · result
- [diag_S2tr_7103248.json](diag_S2tr_7103248.json) · result
- [EVIDENCE_S2tr_v1.txt](EVIDENCE_S2tr_v1.txt) · result
- [guard_diag.json](guard_diag.json) · result
- [quick_s2t_r1.json](quick_s2t_r1.json) · result
- [quick_s2t_r1_guard.json](quick_s2t_r1_guard.json) · result
- [quick_s2t_r2.json](quick_s2t_r2.json) · result
- [quick_s2t_r2_guard.json](quick_s2t_r2_guard.json) · result
- [quick_s2t_r3.json](quick_s2t_r3.json) · result

</details>

## Related work

- shipped in: [Submission #336412: rank-384 stack with own prebind census, accumulator refit and least-recently-used caches](../../../submissions/336412/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-rank-384-stack-census-refit-977-fewer-first-351) · [Copy provenance](../../../PROVENANCE.json)

