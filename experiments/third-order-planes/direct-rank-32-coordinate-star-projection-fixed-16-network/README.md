# Direct rank-32 coordinate-star projection: fixed 16-network increment improves 0.499758%

2026-09-28 · experiment · adopted · measured

A deterministic-column rank32 star projection with one power iteration was added to late common256 and thin-star routing. The original narrow confidence failure was retained, and one predeclared unchanged-source extra-eight panel qualified the component.

Recorded measurement:

    TRAIN8 versus #332517 adjusted-1.352971%,8/8wins,CI[-1.795945%,-0.928311%];incremental TRAIN16 versus common256-0.499758%,12/16wins,CI[-0.826831%,-0.171318%];raw+0.057483%,compute-0.557997%;extra8-0.631778%,5/8wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/teammate_recovery_20260927_v1/late_common256.py](code/teammate_recovery_20260927_v1/late_common256.py)

## Evidence

- [HILL_CLIMB_CURRENT_before_332738.md#L5](../common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md#L5)

<details>
<summary>All result files and records (1)</summary>

- [HILL_CLIMB_CURRENT_before_332738.md](../common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md) · record

</details>

## Related work

- shipped in: [Submission #332738: rank-32 coordinate-star projection](../../../submissions/332738/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-direct-rank-32-coordinate-star-projection-fixed-16-network) · [Copy provenance](../../../PROVENANCE.json)

