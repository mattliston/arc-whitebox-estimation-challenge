# Tree aliases with union push products saved 0.333% bill at flat isolated backend and residual time

2026-10-17T18:18:11Z · experiment · adopted · measured

Unioning inherited and newborn push products with shared tree aliases removed duplicated arithmetic. Removing the rank-48 kernel and its compute wrapper preserved the saving with negligible isolated runtime change; the union entered #336779.

Recorded measurement:

    Rank 100, three pairs: bill -0.333%; backend +0.46/+0.04/-0.24%; residual +0.2/+0.3/+0.3%; requests +25; public bill -0.26%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_v40u.py](code/candidate_v40u.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4898](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4898)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4900](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4900)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4909](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4909)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

## Related work

- shipped in: [Submission #336779: union of inherited and newborn push products: 1.4688380752067338e-09](../../../submissions/336779/README.md)

[Research graph](../../../graph/index.html#node=2026-10-17-tree-aliases-union-push-products-saved-0-333-bill) · [Copy provenance](../../../PROVENANCE.json)

