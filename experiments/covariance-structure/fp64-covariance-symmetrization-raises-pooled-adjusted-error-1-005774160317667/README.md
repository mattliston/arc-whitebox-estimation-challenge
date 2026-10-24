# FP64 covariance symmetrization raises pooled adjusted error to 1.005774160317667

2026-09-20 · experiment · rejected · measured

The full preactivation covariance matrices were symmetrized before consumption while the directed third-cumulant plane remained unchanged.

Recorded measurement:

    One insertion symmetrizes full Cpre/Bpre in FP64 before their consumers.
    The directed K21 plane, final diagonal path, rank384/age4, all learned
    | 35 | 1.870602849007794e-08 | 1.875453353257169e-08 | 1.00259301661 | 1.0028993619 |
    | 100 | 1.842453417527493e-08 | 1.85786822762268e-08 | 1.00836645852 | 1.00867240662 |
    Pooled adjusted ratio: **1.005774160317667**.
    The fixed gate requires both raw/adjusted wins and pooledadjusted<=.99.
    These are two reused ordinary1B development networks, not independent
    Rank35 cold FLOPs: 398080917697 -> 398202552513 (+121634816).
    Rank100 cold FLOPs: 400893063472 -> 401014698288 (+121634816).
    1.06581410364015e-14.
    network repeat/warm-cold100 bytes, first-layer parent mean equality,
    Four predictions total 1565900581246 FLOPs; peak RSS 6073057280 bytes.
    All four guarded stages exit0 and retire; zero added swap.
    Future qualified Phase2 board uploads have the user’s standing preauthorization.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/covariance_symmetry_v1/build.py](code/covariance_symmetry_v1/build.py)
- [code/covariance_symmetry_v1/candidate.py](code/covariance_symmetry_v1/candidate.py)
- [code/covariance_symmetry_v1/finalize.py](code/covariance_symmetry_v1/finalize.py)
- [code/covariance_symmetry_v1/score_dev.py](code/covariance_symmetry_v1/score_dev.py)
- [code/covariance_symmetry_v1/symmetry_fnp.py](code/covariance_symmetry_v1/symmetry_fnp.py)
- [code/covariance_symmetry_v1/unit.py](code/covariance_symmetry_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-fp64-covariance-symmetrization-raises-pooled-adjusted-error-1-005774160317667) · [Copy provenance](../../../PROVENANCE.json)

