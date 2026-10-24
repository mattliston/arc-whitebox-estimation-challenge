# Join rows from 16 top anchors plus conditional Gram-residual selection (layers 10/13): pilot-8 -0.42%, fresh-16 +0.25%

2026-10-15T03:46:45Z · experiment · rejected · measured

A diversity row set for the join (the top 16 anchors, then rows chosen by conditional Gram residual, same ranks, at most +441M FLOPs / +76 requests) won its pilot by a wide margin and lost the 16 fresh networks of the full science; the pooled 24 was negative only because of the 8 out-of-sample networks.

Recorded measurement:

    pilot 8: adjusted -0.422046%, MSE -0.547012%, bill +0.125653% (5/8 lower); full: fresh 16 +0.245301%, OOS 8 -0.692567%, pooled 24 -0.068302%; closed by the registered fresh-16 sign

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_join_conditional16_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_jcr16_HELD_v2.py](code/candidate_jcr16_HELD_v2.py)
- [code/check_jcr16_v2.py](code/check_jcr16_v2.py)
- [code/kernel_jcr16_HELD_v2.py](code/kernel_jcr16_HELD_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3295](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3295)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3299](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3299)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3302](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3302)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3306](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3306)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-join-rows-16-top-anchors-plus-conditional-gram-residual) · [Copy provenance](../../../PROVENANCE.json)

