# Direct star rank 40: lower raw error is outweighed by additional compute

2026-09-28 · experiment · rejected · measured

The direct star projection was increased from rank32 to40 with one power iteration and matching depth-three routes. The global rank step failed the two-case screen.

Recorded measurement:

    TRAIN2 raw-0.359184%,bill+0.404409%,adjusted+0.043659%,1/2wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_star_projection40_v1/candidate.py](code/hill_climb_star_projection40_v1/candidate.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)

## Evidence

- [HILL_CLIMB_before_uvp_bcache_v1.md#L29](../../birth-terms/birth-selector-enlarged-32-units-layers-110-expanded-adjusted/HILL_CLIMB_before_uvp_bcache_v1.md#L29)

<details>
<summary>All result files and records (3)</summary>

- [HILL_CLIMB_before_uvp_bcache_v1.md](../../birth-terms/birth-selector-enlarged-32-units-layers-110-expanded-adjusted/HILL_CLIMB_before_uvp_bcache_v1.md) · record
- [result.json](result.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-direct-star-rank-40-lower-raw-error-outweighed-additional) · [Copy provenance](../../../PROVENANCE.json)

