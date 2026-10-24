# Birth-only threshold 0.05 raises pooled adjusted error 37.804% despite about 9% FLOP savings

2026-09-19 · experiment · rejected · measured

Only the newborn selector changed; active and response floors stayed at 0.01, with all learned parameters fixed.

Recorded measurement:

    # Fixed current-Lazy birth-only0.05 objective: negative
    active/response floors0.01 while using the old fixed0.05 newborn rule saves
    only8.972%/8.741% of cold FLOPs, and loses much more in accuracy:
    | 35 | 1.461342058840 | 0.910280560343 | 1.330231268174 |
    | 100 | 1.562842128598 | 0.912592074658 | 1.426237340501 |
    Pooled own-cold adjusted ratio **1.37803933450579**,
    a37.804% worsening. The predeclared gates required both cost ratios<=.90,
    both adjusted ratios<1, and pooled<=.98. All three fail. Even without the
    selectors, formulas, rank384/age4, Lazy factors, kernels and cache rules stay
    Historical compact_birthonly_v1 remains a separate unlearned negative:
    raw ratios1.3523137457806322/1.2646735350385387 with no metered bill. This
    repeated plane, while changing all-distinct entries (difference norm.90563
    on the toy). Maximum FP64 plane error5.55e-16 and actual FP32-star scaled
    error4.48e-7 satisfy the fixed gates. These local identities did not promise
    future accuracy. Unit v1 reached reporting but JSON serialization rejected
    Unit v2 only casts the report count to Python int and uses a new output
    path. Candidate and numerical checks are unchanged; v2 passes.
    All four full16-layer calls complete, finite/nonnegativefloat32. Repeated35
    and warm/cold100 outputs are SHA-identical. The first3 rows also match the
    parent byte-for-byte on both nets (the gate required2). Source/parity/output
    seals precede the existing1B reference accesses. There is no4M noise
    Cold35 bill362365320823F; warm100355151323843F; repeat35341761068597F;
    cold100365851832510F. Total1425129545773 billed
    prediction FLOPs. Native walls29.3634/26.7436/27.6859/27.2743s. Five guard
    wall114.794296s, peak5976997888B,

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/birth05_objective_v1/build.py](code/birth05_objective_v1/build.py)
- [code/birth05_objective_v1/candidate.py](code/birth05_objective_v1/candidate.py)
- [code/birth05_objective_v1/kernel.py](code/birth05_objective_v1/kernel.py)
- [code/birth05_objective_v1/score_dev.py](code/birth05_objective_v1/score_dev.py)
- [code/birth05_objective_v1/score_dev_v2.py](code/birth05_objective_v1/score_dev_v2.py)
- [code/birth05_objective_v1/unit.py](code/birth05_objective_v1/unit.py)
- [code/birth05_objective_v1/unit_v2.py](code/birth05_objective_v1/unit_v2.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L1](NEGATIVE_RESULT.md#L1)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [REEXAMINATION_INDEX_20260919.md#L1437](REEXAMINATION_INDEX_20260919.md#L1437)
- [REEXAMINATION_INDEX_20260919.md#L1439](REEXAMINATION_INDEX_20260919.md#L1439)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result
- [REEXAMINATION_INDEX_20260919.md](REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-birth-only-threshold-0-05-raises-pooled-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

