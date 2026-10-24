# Late common-factor rank 192: adjusted score loses 0.447604% to rank 256

2026-09-28 · experiment · rejected · measured

The adjacent rank192 candidate received its frozen six-network extension. Its gain against the older baseline was uncertain and it lost against retained rank256.

Recorded measurement:

    TRAIN8 versus #332517 adjusted-0.552965%,5/8wins,CI crosses zero;versus rank256+0.447604%,3/8wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/teammate_recovery_20260927_v1/build.py](code/teammate_recovery_20260927_v1/build.py)
- [code/teammate_recovery_20260927_v1/check_edge_projection_exact.py](code/teammate_recovery_20260927_v1/check_edge_projection_exact.py)
- [code/teammate_recovery_20260927_v1/late_common192.py](code/teammate_recovery_20260927_v1/late_common192.py)
- [code/teammate_recovery_20260927_v1/late_common256.py](code/teammate_recovery_20260927_v1/late_common256.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/candidate_thin_star_route_v1.py](code/candidate_thin_star_route_v1.py)

## Evidence

- [hill_climb_reset_20260928_v1.md#L22](../../covariance-structure/explicit-covariance-symmetrization-adjusted-gain-0-516381-stays-below/hill_climb_reset_20260928_v1.md#L22)

<details>
<summary>All result files and records (2)</summary>

- [hill_climb_reset_20260928_v1.md](../../covariance-structure/explicit-covariance-symmetrization-adjusted-gain-0-516381-stays-below/hill_climb_reset_20260928_v1.md) · record
- [result.json](result.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-late-common-factor-rank-192-adjusted-score-loses-0) · [Copy provenance](../../../PROVENANCE.json)

