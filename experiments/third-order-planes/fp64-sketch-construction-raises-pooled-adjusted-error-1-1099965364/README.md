# FP64 sketch construction raises pooled adjusted error to 1.1099965364 while raw error is nearly unchanged

2026-09-20 · experiment · rejected · measured

The unperturbed predictions from the precision intervention were scored with their complete cold bills.

Recorded measurement:

    # Frozen FP64 sketch endpoint: negative adjusted objective
    from fp64_sketch_response_v1. No new estimator calls or fitting occurred.
    finite nonnegative outputs and first5 parent bytes pass qualification.
    | 36 | 0.999504330949 | 1.10880976698 | 1.10826016429 |
    | 38 | 1.00303275784 | 1.10892696696 | 1.11229007392 |
    Pooled raw ratio1.00103634846; pooled adjusted ratio
    1.1099965364. The frozen candidate is about11.0% WORSE
    Raw error is essentially unchanged; the added10.9% bill is not repaid.
    downloads or GPU. Official331684 remains3.5844144212297072e-9; goal active.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fp64_sketch_endpoint_v1/finalize.py](code/fp64_sketch_endpoint_v1/finalize.py)
- [code/fp64_sketch_endpoint_v1/prepare.py](code/fp64_sketch_endpoint_v1/prepare.py)
- [code/fp64_sketch_endpoint_v1/score.py](code/fp64_sketch_endpoint_v1/score.py)
- [code/fp64_sketch_response_v1/candidate.py](code/fp64_sketch_response_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L1](RESULT.md#L1)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L10](RESULT.md#L10)
- [REEXAMINATION_INDEX_20260919.md#L773](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L773)
- [REEXAMINATION_INDEX_20260919.md#L778](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L778)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [score.json](score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-fp64-sketch-construction-raises-pooled-adjusted-error-1-1099965364) · [Copy provenance](../../../PROVENANCE.json)

