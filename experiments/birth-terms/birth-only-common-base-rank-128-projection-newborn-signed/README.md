# Birth-only common-base rank-128 projection of newborn signed pairs: final raw error x2.623870 / x2.298309 (rejected)

2026-09-19 · experiment · rejected · measured

The common base of each newborn signed pair (columns B[:,j]+t_j e_j and B[:,j]+s_j e_j) was projected at birth to rank 128 for births 4..14 while the coordinate-difference leg was kept exact and the repeated-plane change absorbed in the existing star reset; an offline unpriced SVD measured accuracy only. Even a free projection with the optimistic 11.6% whole-card saving could not compensate the error, so this projection family was closed without a carrier rewrite.

Recorded measurement:

    final raw error ratio 2.623870 on net 35, 2.298309 on net 100, pooled 2.462524; young-push bill is 31% of I95 so an optimistic 37.5% cut there is 11.6% of the total

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/common_projection_probe.py](code/codex_resume_20260919_v1/common_projection_probe.py)
- [code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py](code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py)

## Evidence

- [RECEIPTS.md#L91](RECEIPTS.md#L91)
- [RECEIPTS.md#L93](RECEIPTS.md#L93)
- [COMMON_LEG_SCREEN.md#L17](COMMON_LEG_SCREEN.md#L17)
- [COMMON_LEG_SCREEN.md#L42](COMMON_LEG_SCREEN.md#L42)

<details>
<summary>All result files and records (3)</summary>

- [COMMON_LEG_SCREEN.md](COMMON_LEG_SCREEN.md) · record
- [common35_100_score.json](common35_100_score.json) · result
- [RECEIPTS.md](RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-birth-only-common-base-rank-128-projection-newborn-signed) · [Copy provenance](../../../PROVENANCE.json)

