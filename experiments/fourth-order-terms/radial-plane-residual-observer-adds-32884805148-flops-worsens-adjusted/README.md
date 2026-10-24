# Radial-plane residual observer adds 32884805148 FLOPs and worsens adjusted error: ratio 1.082647443018

2026-09-21 · experiment · rejected · measured

A separate observer propagated the fourth-plane residual through next-layer weights and Gaussian gains at unit coefficient without changing the parent physical state.

Recorded measurement:

    symmetric radial projection from Q, casts that residual D to FP32, and
    only the observer's mean shift before the original nonnegative FP32 cast.
    | Existing 1B network | Raw error ratio | Complete cold cost ratio | Adjusted error ratio |
    | 35 | 1.002188913427 | 1.079537813020 | 1.081900827833 |
    | 100 | 1.004095817722 | 1.078999727451 | 1.083419113657 |
    Pooled adjusted ratio: **1.082647443018**. The fixed
    gate requires both adjusted ratios <1 and pooled <=.99. Passed: False.
    Cold parent bills are 413448696909 /
    416264792412 F; candidate bills are
    446333502057 / 449149597560 F.
    predict. Extra observer work is 32884805148 F per call.
    The cost multiplier is each full cold bill divided by 2^41, floored at .1.
    FNP bytes and bills. The zero double trace applies before FP32 rounding.
    Disabled cold35 exactly reproduces all 16 parent rows and the complete bill.
    Enabled35/100/35 and independent cold100 each preserve all 16 underlying
    shifts repeat exactly for repeated35 and warm/cold100. First output rows,
    finite/nonnegative FP32 contract and numeric-cache retirement checks pass.
    each cover 60 layer events. Their maximum relative errors are
    2.90872266099e-08, 1.51001081694e-06 and
    1.4913018349e-06, within frozen 2e-6 / 3e-5 / 3e-5 limits.
    Five full native calls total 2174583804499 F. Six serial
    guarded groups retired; controller 77086 is terminal. Peak sampled group
    RSS is 6718504960 B and self-report is
    6718455808 B; added swap is zero. Fast
    2.361 MB. No new samples, fits, downloads or GPU calls.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/candidate.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/candidate.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/common.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/common.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/finalize.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/finalize.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/fragment.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/fragment.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/helper.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/helper.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/native.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/native.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/preflight.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/preflight.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/prepare.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/prepare.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/qualify.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/qualify.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/run.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/run.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/score.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/score.py)
- [code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/unit.py](code/codex_resume_20260919_v1/radial_plane_mean_observer_v1/unit.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L13](RESULT.md#L13)
- [RESULT.md#L15](RESULT.md#L15)
- [REEXAMINATION_INDEX_20260919.md#L671](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L671)
- [REEXAMINATION_INDEX_20260919.md#L674](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L674)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [score.json](score.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-radial-plane-residual-observer-adds-32884805148-flops-worsens-adjusted) · [Copy provenance](../../../PROVENANCE.json)

