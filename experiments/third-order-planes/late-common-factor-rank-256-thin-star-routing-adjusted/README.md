# Late common-factor rank 256 with thin star routing: adjusted training score improves 0.996111%

2026-09-28 · experiment · adopted · measured

The unchanged late common-factor rank256 candidate received the predeclared six-network follow-up after its earlier small-screen gate failure. Its fixed eight-network gain qualified it as the next experimental parent.

Recorded measurement:

    TRAIN8 versus #332517 adjusted-0.996111%,8/8wins,95%CI[-1.32661%,-0.67454%];increment versus thin routing-0.425085%,7/8wins,CI[-0.82760%,-0.05768%].

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/teammate_recovery_20260927_v1/build.py](code/teammate_recovery_20260927_v1/build.py)
- [code/teammate_recovery_20260927_v1/check_edge_projection_exact.py](code/teammate_recovery_20260927_v1/check_edge_projection_exact.py)
- [code/teammate_recovery_20260927_v1/late_common192.py](code/teammate_recovery_20260927_v1/late_common192.py)
- [code/teammate_recovery_20260927_v1/late_common256.py](code/teammate_recovery_20260927_v1/late_common256.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/candidate_thin_star_route_v1.py](code/candidate_thin_star_route_v1.py)

## Evidence

- [hill_climb_reset_20260928_v1.md#L21](../../covariance-structure/explicit-covariance-symmetrization-adjusted-gain-0-516381-stays-below/hill_climb_reset_20260928_v1.md#L21)

<details>
<summary>All result files and records (2)</summary>

- [hill_climb_reset_20260928_v1.md](../../covariance-structure/explicit-covariance-symmetrization-adjusted-gain-0-516381-stays-below/hill_climb_reset_20260928_v1.md) · record
- [result.json](result.json) · result

</details>

## Related work

- shipped in: [Submission #332738: rank-32 coordinate-star projection](../../../submissions/332738/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-late-common-factor-rank-256-thin-star-routing-adjusted) · [Copy provenance](../../../PROVENANCE.json)

