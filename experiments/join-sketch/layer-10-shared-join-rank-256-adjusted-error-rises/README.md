# Layer-10 shared join rank 256: adjusted error rises 9.132025366%

2026-09-28 · experiment · rejected · measured

Only the layer10 shared join rank was reduced to256. All outputs through layer9 were exact, but the native endpoint lost badly.

Recorded measurement:

    TRAIN2 adjusted+9.132025366%,raw+10.643806923%,bill-1.368019634%,0/2wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_join10_rank256_v1/candidate.py](code/hill_climb_join10_rank256_v1/candidate.py)
- [code/hill_climb_join10_rank256_v1/build.py](code/hill_climb_join10_rank256_v1/build.py)
- [code/hill_climb_join10_rank256_v1/fragment.py](code/hill_climb_join10_rank256_v1/fragment.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)

## Evidence

- [HILL_CLIMB_before_hk_trace_closeout_v1.md#L55](HILL_CLIMB_before_hk_trace_closeout_v1.md#L55)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_before_hk_trace_closeout_v1.md](HILL_CLIMB_before_hk_trace_closeout_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-layer-10-shared-join-rank-256-adjusted-error-rises) · [Copy provenance](../../../PROVENANCE.json)

