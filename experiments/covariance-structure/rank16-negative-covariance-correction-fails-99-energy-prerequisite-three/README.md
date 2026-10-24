# Rank16 negative-covariance correction fails 99% energy prerequisite on three of fourteen cells

2026-09-21 · experiment · rejected · measured

The spectral screen stopped before opening accuracy targets because sixteen modes did not retain the required fraction of negative energy.

Recorded measurement:

    The fixed rank16 action fails its spectral-energy prerequisite on both
    |Rank|Post layer|Negative energy retained by16 modes|
    |35|10|0.987867190709|
    |35|11|0.984375375637|
    |100|13|0.972735234286|
    Every cell required at least0.99. No rank, threshold, diagonal scaling,
    8M covariance labels or ordinary1B means were decoded. No native candidate,
    6.44328e-08 to 3.77206e-07 of the actual covariance
    norm over these cells. This describes the current saved331684 trajectory;
    negative counts range32..129; signs very near zero can be sensitive to
    Spectra agree within 2.55562e-15 relative, and retained-energy fractions
    within 3.69774e-11 absolute. Selected factors satisfy the original
    within8.13e-11 relative.
    2.66523e-07 to 1.72375e-06 RMS for the immediate Gaussian mean
    standard positive-semidefinite projection construction ([Higham,1988](https://eprints.maths.manchester.ac.uk/363/)).
    diagonal and keeps at most16 modes; it is not that full projection or the
    444219392 bytes; self-reported peak
    444219392 bytes. Artifacts before closeout
    total 2033899 bytes. No new forward
    Official331684 remains adjusted3.5844144212297072e-9,100/100 complete,

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/negative_covariance_modes_v1/audit.py](code/negative_covariance_modes_v1/audit.py)
- [code/negative_covariance_modes_v1/common.py](code/negative_covariance_modes_v1/common.py)
- [code/negative_covariance_modes_v1/finalize.py](code/negative_covariance_modes_v1/finalize.py)
- [code/negative_covariance_modes_v1/geometry.py](code/negative_covariance_modes_v1/geometry.py)
- [code/negative_covariance_modes_v1/prepare.py](code/negative_covariance_modes_v1/prepare.py)
- [code/negative_covariance_modes_v1/run.py](code/negative_covariance_modes_v1/run.py)
- [code/negative_covariance_modes_v1/score.py](code/negative_covariance_modes_v1/score.py)
- [code/negative_covariance_modes_v1/spectrum.py](code/negative_covariance_modes_v1/spectrum.py)
- [code/negative_covariance_modes_v1/unit.py](code/negative_covariance_modes_v1/unit.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L9](RESULT.md#L9)
- [REEXAMINATION_INDEX_20260919.md#L731](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L731)
- [REEXAMINATION_INDEX_20260919.md#L734](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L734)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-rank16-negative-covariance-correction-fails-99-energy-prerequisite-three) · [Copy provenance](../../../PROVENANCE.json)

