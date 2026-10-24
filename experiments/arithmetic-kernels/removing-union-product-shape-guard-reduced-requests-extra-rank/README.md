# Removing the union-product shape guard reduced requests; the extra rank-48 kernel had no covered shapes

2026-10-17T19:03:05Z · experiment · adopted · measured

Union of inherited and newborn push products was enabled for all roster shapes instead of a fixed coverage list. It reduced request count on the clean store comparison; adding the rank-48 kernel changed no bill because its two shapes did not occur.

Recorded measurement:

    Clean 32-rank bill -0.170% (SE .033, range -0.55..+0.21); requests -682; raw U -0.002% (SE .07, 17/31 lower); rank-48 kernel has identical bill on 8 ranks.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_v40ua.py](code/candidate_v40ua.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4905](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4905)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4906](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4906)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4908](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4908)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4912](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4912)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result
- [GRADE_STORE_v1-2d13eaa7.json](GRADE_STORE_v1-2d13eaa7.json) · result

</details>

## Related work

- shipped in: [Submission #336831: union push products for all roster shapes: three residual failures](../../../submissions/336831/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-removing-union-product-shape-guard-reduced-requests-extra-rank) · [Copy provenance](../../../PROVENANCE.json)

