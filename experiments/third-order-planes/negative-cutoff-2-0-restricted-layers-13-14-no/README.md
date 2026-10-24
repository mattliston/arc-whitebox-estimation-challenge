# Negative cutoff 2.0 restricted to layers 13 and 14: no compute saving and worse error

2026-09-28 · experiment · rejected · measured

The negative-tail cutoff was changed only at layers13 and14. The complete bill did not change while both scores worsened.

Recorded measurement:

    TRAIN2 adjusted+0.740673%,0/2wins;compute unchanged.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_late_negative20_v1/candidate.py](code/hill_climb_late_negative20_v1/candidate.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT_before_332738.md#L20](../common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md#L20)

<details>
<summary>All result files and records (1)</summary>

- [HILL_CLIMB_CURRENT_before_332738.md](../common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-negative-cutoff-2-0-restricted-layers-13-14-no) · [Copy provenance](../../../PROVENANCE.json)

