# Sketch condition number grows from 5,676 to 3,427,201; FP32 projector error reaches 0.212442

2026-09-20 · finding · informational · measured

Unchanged-state replay isolated range-sketch arithmetic sensitivity and established that QR already used widened internal arithmetic.

Recorded measurement:

    Five rank38 cold replays reproduce all16 original output bytes and all full
    all saved trace arrays exactly. Three ownership/QR/Gram fixtures pass. All40
    actual native QR results equal FP64 refactorization rounded toFP32 exactly.
    Frozen-current-input FP64 sketch reconstruction differs from the native
    sketch by3.91e-6..5.56e-6 relative on the base trajectory. Native condition
    numbers grow from5,676 at layer5 to3,427,201 at layer14. Corresponding
    relative Frobenius projector differences grow from0.000887 to0.212442.
    At the first changed sketch (layer10), native centered projector response
    norm is0.581818, versus0.000165167 for the independently reconstructed
    same-input FP64 sketch. Their forward/center discrepancies are1.80812 and
    0.0697395, respectively. Later FP64 references still inherit perturbed
    and actual own cost. Merely declaring QR64 is not the proposed mechanism:
    NumPy already widens QR internally, while the sketch producer isFP32.
    Layers5..9 have zero cross-perturbation sketch response. In analysis.json the
    field base_sketch_rounding_over_centered_response uses a1e-300 numerical
    Seven guarded groups retired;5 full calls,2.072718626895TF including
    159,583,360 diagnostic copyF; peak sampled6.342GB, zero added swap.
    About639MB artifacts; no new samples, downloads, GPU or training.
    Official331684 stays3.5844144212297072e-9; remaining milestones unmet.

## Programs

- [analyze.py](analyze.py)
- [capture.py](capture.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [prepare.py](prepare.py)
- [run.py](run.py)
- [../../../experiments/third-order-planes/fp64-range-sketch-construction-improves-sensitivity-but-fails-all/code/shared_sketch_roundoff_diagnostic_v1/snapshot.py](../../../experiments/third-order-planes/fp64-range-sketch-construction-improves-sensitivity-but-fails-all/code/shared_sketch_roundoff_diagnostic_v1/snapshot.py)
- [unit.py](unit.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L8](RESULT.md#L8)
- [REEXAMINATION_INDEX_20260919.md#L773](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L773)
- [REEXAMINATION_INDEX_20260919.md#L776](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L776)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-sketch-condition-number-grows-5-676-3-427-201) · [Copy provenance](../../../PROVENANCE.json)

