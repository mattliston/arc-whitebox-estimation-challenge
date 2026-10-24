# Frozen local mean-innovation correction worsens pooled adjusted error 4.281639440%

2026-09-20 · experiment · rejected · measured

The fixed amplitude-one correction at layers8-15 used current live features and autonomous state propagation rather than a tangent forecast.

Recorded measurement:

    The fixed amplitude-one correction at layers8..15 passed numerical/native
    by 4.281639440%. Ratios35/100 are
    1.04281172612066/1.04282122189414.
    Candidate adjusted scores are 3.49845049677837e-09 and
    3.38310657874452e-09, versus the exact full-lower parent
    3.35482466215919e-09/3.24418654675976e-09.
    Compute increased only about0.014%; the failure is primarily raw accuracy.
    editing restores the parent byte for byte. Disabled35/100/35 has all16 output
    rows and complete bills exact. Enabled35/100/35+cold100 has first8 rows exact,
    exact own repeats and warm/cold100 output/feature/source bytes. All32 wide
    local-source evaluations match the original FP32 NumPy model exactly; maximum
    independent FP64 relative discrepancy is
    5.91064227153e-07.
    cold and warm helper costs. Separate module initialization measured0F; frozen
    parameter materialization measured7036F. The v3 test explicitly proves cold
    The old TRAIN loss.876624 failed its original.50 gate and remains failed. This
    failures), peak6619250688B, zero added swap. Seven
    full calls/2820547332610F, twelve small source calls/
    1460418F,32 wide reference cells; no fit/new samples/download/GPU work. The
    official bank331672 remains3.6316074532884024e-9; readout331676 is separate

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_lower_mean_innovation_native_v1/build.py](code/full_lower_mean_innovation_native_v1/build.py)
- [code/full_lower_mean_innovation_native_v1/candidate.py](code/full_lower_mean_innovation_native_v1/candidate.py)
- [code/full_lower_mean_innovation_native_v1/common.py](code/full_lower_mean_innovation_native_v1/common.py)
- [code/full_lower_mean_innovation_native_v1/finalize.py](code/full_lower_mean_innovation_native_v1/finalize.py)
- [code/full_lower_mean_innovation_native_v1/native.py](code/full_lower_mean_innovation_native_v1/native.py)
- [code/full_lower_mean_innovation_native_v1/native_v2.py](code/full_lower_mean_innovation_native_v1/native_v2.py)
- [code/full_lower_mean_innovation_native_v1/native_v3.py](code/full_lower_mean_innovation_native_v1/native_v3.py)
- [code/full_lower_mean_innovation_native_v1/qualify.py](code/full_lower_mean_innovation_native_v1/qualify.py)
- [code/full_lower_mean_innovation_native_v1/run_development.py](code/full_lower_mean_innovation_native_v1/run_development.py)
- [code/full_lower_mean_innovation_native_v1/run_development_v2.py](code/full_lower_mean_innovation_native_v1/run_development_v2.py)
- [code/full_lower_mean_innovation_native_v1/run_development_v3.py](code/full_lower_mean_innovation_native_v1/run_development_v3.py)
- [code/full_lower_mean_innovation_native_v1/score.py](code/full_lower_mean_innovation_native_v1/score.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/full_lower_mean_innovation_native_v1/unit.py](code/full_lower_mean_innovation_native_v1/unit.py)
- [code/full_lower_mean_innovation_native_v1/unit_v2.py](code/full_lower_mean_innovation_native_v1/unit_v2.py)
- [code/full_lower_mean_innovation_native_v1/unit_v3.py](code/full_lower_mean_innovation_native_v1/unit_v3.py)
- [code/full_lower_moment_panel_v1/candidate.py](code/full_lower_moment_panel_v1/candidate.py)
- [code/lazy_mean_innovation_observer_v1/model.py](code/lazy_mean_innovation_observer_v1/model.py)
- [code/run_guarded.py](code/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L7](RESULT.md#L7)
- [REEXAMINATION_INDEX_20260919.md#L860](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L860)
- [REEXAMINATION_INDEX_20260919.md#L864](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L864)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [score.json](score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-frozen-local-mean-innovation-correction-worsens-pooled-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

