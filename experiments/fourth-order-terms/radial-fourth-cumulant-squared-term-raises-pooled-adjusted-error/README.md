# Radial fourth-cumulant squared term raises pooled adjusted error 1.416381%

2026-09-19 · experiment · rejected · measured

The connected density contribution updated marginal raw moments and three pair-covariance channels consistently through correlation degree two.

Recorded measurement:

    Phase 2 only, 2026-09-19. This new physical-map experiment is complete.
    The candidate appends only the radial L4-squared density contribution to
    Weight5, consistently updating marginal raw moments and the three pair
    This is not the older, already failed L3-squared matched-fit experiment.
    | Network | Weight5 raw MSE | R44 raw MSE | R44/Weight5 adjusted |
    |35|1.871255439572e-08|1.874321294385e-08|1.010914889932|
    |100|1.854682334048e-08|1.869797756672e-08|1.017417471692|
    Pooled adjusted ratio **1.014163812250**, worse by
    1.416381%. Both individual raw errors worsen.
    The fixed both-improve and pooled<=.99 gates fail. No fitting, coefficient,
    Both comparisons use complete independently cold bills: R44
    409303658487F /412331831396F versus Weight5
    405547750335F /408575923244F. Extra work is exactly 3755908152F per call,
    against existing 1B targets on two reused development networks, not the
    noise-corrected14 panel or a new official score.
    Independent formal moment-cumulant recurrence passes25 polynomial
    covariance cases with exact rational coefficients. Actual FP32 helper
    relative error 1.000148808733e-07; marginal Hermite checks
    max 2.116023005763e-16. Null, constant-function,
    Four full predictions (35/100/35 + cold100) were sealed before targets.
    Own repeat, warm/cold100 and original first-layer bytes match exactly;
    all16 served rows are finite and nonnegative. This is not a production
    client/domain qualification. Total billed prediction work 1609390619726F.
    Guarded peak group RSS 6175752192B; conservative maximum including native
    self-report 6247743488B. Zero added swap, all owned groups retired.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/radial_fourth_squared_v1/build.py](code/radial_fourth_squared_v1/build.py)
- [code/radial_fourth_squared_v1/candidate.py](code/radial_fourth_squared_v1/candidate.py)
- [code/radial_fourth_squared_v1/kernel.py](code/radial_fourth_squared_v1/kernel.py)
- [code/radial_fourth_squared_v1/score_dev.py](code/radial_fourth_squared_v1/score_dev.py)
- [code/radial_fourth_squared_v1/unit.py](code/radial_fourth_squared_v1/unit.py)
- [code/radial_fourth_squared_v1/unit_stdlib_v2.py](code/radial_fourth_squared_v1/unit_stdlib_v2.py)
- [code/weight5_reuse_v1/candidate.py](code/weight5_reuse_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L8](NEGATIVE_RESULT.md#L8)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-radial-fourth-cumulant-squared-term-raises-pooled-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

