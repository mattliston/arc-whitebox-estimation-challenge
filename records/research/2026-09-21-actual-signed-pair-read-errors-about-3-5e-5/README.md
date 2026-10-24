# Actual signed-pair read errors are about 3.5e-5 of the full plane; formal covariance effect about 1e-6

2026-09-21 · finding · informational · measured

Independent streamed FP64 references measured rounding on unchanged factor state and separated local numerical discrepancy from final-score improvement.

Recorded measurement:

    effective read discrepancies exceed 3e-5 of the complete off-diagonal K21
    | Network | Maximum error / full K21 | Pooled error / full K21 | Pooled formal covariance response | Maximum formal covariance response |
    | 35 | 5.74941407986e-05 | 3.51190087459e-05 | 1.04917298571e-06 | 1.56654710964e-06 |
    | 100 | 5.33202585774e-05 | 3.4213247263e-05 | 1.0825297675e-06 | 1.45958136916e-06 |
    Each network flags layers 2 and 4 through 14; layers 1 and 3 remain below
    the fixed threshold. All 14 complete reads per network were observed,
    CP read itself, the maximum effective discrepancies are 7.85736010596e-5
    and 7.51850321734e-5. The reference preserves the declared old factorization
    FP32 arithmetic within a read. Diagonal entries are excluded because the
    actual consumer overwrites them through its independent FP64 scalar path.
    The FP64 reference uses streamed 64-row products. A second reference sums
    257-column contractions before the same old maps. Maximum relative reference
    disagreements are 2.63105946172e-15 and 2.41320554950e-15, at most
    3.71209273476e-11 of the measured read-error norms. Six tiny fixtures also
    edge-correlation terms through degree2 and FP32 series arithmetic. The old
    degree0 response was incorrect. RESPONSE_CORRECTION.md documents the exact
    references, fixtures, native controls and the 3e-5 numerical flag did not
    The corrected local covariance response uses the actual series' formal FP64
    algebra with original scalar inputs, including correlation degrees0..2.
    Its independent unbatched FP64 finite-difference certificate has maximum
    absolute error 1.27393755267e-18, below the original 1e-12 algebra tolerance.
    The actual rounded FP32 consumer's unit response differs by norm
    2.19303139269e-9, below its separately declared 2.15110744451e-8 rounding
    it is not an exact derivative of FP32 rounding or the learned autonomous
    All 16 predictions and complete cold bills equal the original bank controls

## Programs

- [admit_repaired.py](admit_repaired.py)
- [analyze.py](analyze.py)
- [analyze_v2.py](analyze_v2.py)
- [capture.py](capture.py)
- [capture_v2.py](capture_v2.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [preflight.py](preflight.py)
- [prepare.py](prepare.py)
- [reference.py](reference.py)
- [reference_v2.py](reference_v2.py)
- [run.py](run.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [run_v2.py](run_v2.py)
- [unit.py](unit.py)
- [unit_v2.py](unit_v2.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L13](RESULT.md#L13)
- [REEXAMINATION_INDEX_20260919.md#L545](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L545)
- [REEXAMINATION_INDEX_20260919.md#L548](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L548)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-actual-signed-pair-read-errors-about-3-5e-5) · [Copy provenance](../../../PROVENANCE.json)

