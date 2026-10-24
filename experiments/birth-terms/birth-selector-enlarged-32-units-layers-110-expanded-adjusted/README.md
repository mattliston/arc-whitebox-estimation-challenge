# Birth selector enlarged by 32 units at layers 1–10: expanded adjusted error rises 1.046220%

2026-09-28 · experiment · rejected · measured

One32-unit birth block was added at layers1–10 only. The initial two-case gain failed the fixed16-network expansion.

Recorded measurement:

    TRAIN2 adjusted-0.245412%,raw-1.789964%,bill+1.568523%,2wins;TRAIN16 adjusted+1.046220%,4/16wins,CI[+0.516582,+1.578596],raw-0.703650%,bill+1.752359%;additional14+1.206259%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_birth_plus32_v1/candidate.py](code/hill_climb_birth_plus32_v1/candidate.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)

## Evidence

- [HILL_CLIMB_before_uvp_bcache_v1.md#L10](HILL_CLIMB_before_uvp_bcache_v1.md#L10)

<details>
<summary>All result files and records (3)</summary>

- [HILL_CLIMB_before_uvp_bcache_v1.md](HILL_CLIMB_before_uvp_bcache_v1.md) · record
- [result.json](result.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-birth-selector-enlarged-32-units-layers-110-expanded-adjusted) · [Copy provenance](../../../PROVENANCE.json)

