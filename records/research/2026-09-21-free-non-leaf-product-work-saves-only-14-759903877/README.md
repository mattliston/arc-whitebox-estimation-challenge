# Free non-leaf product work saves only 14.759903877% and 13.690445234%, below 16.304320666% gate

2026-09-21 · finding · rejected · derived

The optimistic bound forgave all matrix-product overhead while retaining every existing leaf multiplication and all work outside products.

Recorded measurement:

    at most **14.759903877% on35 and13.690445234% on100**. Both are below the
    fixed **16.304320666%** uniform planning cut associated with3e-9 at unchanged
    | 35 | 413448696909 | 320637683249 | 61024630243 | 352424066666 |
    | 100 | 416264792412 | 327518800730 | 56988503433 | 359276288979 |
    The saved identity traces contain590 input occurrences each:428 exact
    classes,162 occurrences after the first; granting transpose equivalence
    gives383 classes and207 subsequent occurrences. Empty operands are included.
    This establishes input repetition even though all295 complete products are
    26,911,117,884 and25,120,389,238 FLOPs; even those include unique-input work.
    The existing W5 reuse is already reflected in the recorded bills.
    All590 ordered product records agree between the original cost census and
    restarted. The initial preparation script encountered checkpoint85's null
    Both guarded groups retired; controller3037 is terminal. No new model calls,
    RSS was56852480 bytes; the sparse0.5s supervisor
    samples reached1376256 bytes. Zero added
    swap. About0.065 MB of new
    Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
    100/100 complete,zero failures.4e-9 is banked;3/2/~1e-9 remain unmet and the

## Programs

- [analyze.py](analyze.py)
- [audit.py](audit.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [preflight.py](preflight.py)
- [prepare.py](prepare.py)
- [prepare_before_checkpoint_schema_fix.py](prepare_before_checkpoint_schema_fix.py)
- [run.py](run.py)
- [../2026-09-21-current-295-product-census-attributes-79-0853-80-2033/census.py](../2026-09-21-current-295-product-census-attributes-79-0853-80-2033/census.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L13](RESULT.md#L13)
- [REEXAMINATION_INDEX_20260919.md#L582](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L582)
- [REEXAMINATION_INDEX_20260919.md#L585](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L585)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-free-non-leaf-product-work-saves-only-14-759903877) · [Copy provenance](../../../PROVENANCE.json)

