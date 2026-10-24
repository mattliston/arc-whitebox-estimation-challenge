# Consumer-direction star residual repair: fixed16 gain has confidence crossing zero

2026-09-28 · experiment · rejected · measured

A consumer-direction residual repair improved the initial two cases but failed the fixed16-network uncertainty gate. An immutable-array augmented-assignment runtime failure was preserved before a mechanically repaired version ran.

Recorded measurement:

    TRAIN2 adjusted-.431835%;TRAIN16-.328423%,10wins,CI[-.816288,+.159845],additional14-.31561%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_star_consumer_residual_v2/candidate.py](code/hill_climb_star_consumer_residual_v2/candidate.py)
- [code/hill_climb_star_consumer_residual_v2/build.py](code/hill_climb_star_consumer_residual_v2/build.py)
- [code/hill_climb_star_consumer_residual_v1/candidate.py](code/hill_climb_star_consumer_residual_v1/candidate.py)
- [code/hill_climb_star_consumer_residual_v2/assess16.py](code/hill_climb_star_consumer_residual_v2/assess16.py)
- [code/hill_climb_star_consumer_residual_v2/assess_screen.py](code/hill_climb_star_consumer_residual_v2/assess_screen.py)
- [code/hill_climb_star_consumer_residual_v2/fragment.py](code/hill_climb_star_consumer_residual_v2/fragment.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [HILL_CLIMB_before_birth_screens_v1.md#L25](HILL_CLIMB_before_birth_screens_v1.md#L25)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_before_birth_screens_v1.md](HILL_CLIMB_before_birth_screens_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [train16_assessment.json](train16_assessment.json) · result
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-consumer-direction-star-residual-repair-fixed16-gain-has-confidence) · [Copy provenance](../../../PROVENANCE.json)

