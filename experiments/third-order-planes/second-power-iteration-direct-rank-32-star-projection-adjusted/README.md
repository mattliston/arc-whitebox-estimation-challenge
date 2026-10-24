# Second power iteration in direct rank-32 star projection: adjusted error rises 0.0826773197%

2026-09-28 · experiment · rejected · measured

The retained exact bundle gained only a second power iteration in the rank32 star projection. The two-case screen failed its both-win gate.

Recorded measurement:

    TRAIN2 adjusted+.0826773197%,raw-.1508522185%,bill+.2257281468%,1/2wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_star32_q2_v1/candidate.py](code/hill_climb_star32_q2_v1/candidate.py)
- [code/hill_climb_star32_q2_v1/build.py](code/hill_climb_star32_q2_v1/build.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_star32_q2_v1/assess.py](code/hill_climb_star32_q2_v1/assess.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)

## Evidence

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L7](../canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L7)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md](../canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-second-power-iteration-direct-rank-32-star-projection-adjusted) · [Copy provenance](../../../PROVENANCE.json)

