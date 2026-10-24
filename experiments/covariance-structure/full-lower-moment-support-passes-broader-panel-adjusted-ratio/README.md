# Full lower-moment support passes broader panel: adjusted ratio 0.966649340 and 8/8 confirmation wins

2026-09-20 · experiment · adopted · measured

The preserved full covariance, coordinate-star and trace-support candidate passed broader scoring despite the earlier stricter two-network gate failure.

Recorded measurement:

    Lazy #331599 at adjusted3.8164492657795035e-9; no new upload occurred.
    | public8, noise-corrected4M | 4.03568836827e-09 | 4.21698202966e-09 | 0.957008671 | 0.928372805 | 6 |
    | historical_fresh6, noise-corrected4M | 3.22666892299e-09 | 3.2819169879e-09 | 0.983165916 | 0.952870326 | 3 |
    | all14, noise-corrected4M | 3.68896574886e-09 | 3.81623986891e-09 | 0.966649340 | 0.937415832 | 9 |
    | Confirmation8, ordinary1B | 3.68998061953e-09 | 3.85884272688e-09 | 0.956240220 | 0.925855941 | 8 |
    The twelve new panel rows alone have adjusted ratio0.9661403303943533.
    The original development5%-per-network gate remains failed. This study explicitly
    star and trace support are a coupled intervention; FP32 summation order also
    two prefix rows were byte-identical on every network. score_v2 records per-layer
    Actual NumPy-free client35/100/35 output bytes and FLOP bills match native calls
    exactly, and repeated35 is exact. The original runtime gate FAILED: candidate
    47.0621/43.9091/44.8293s versus historical Lazy40.3462/36.8842/37.9341s.
    Lazy outputs, bills and request counts, but took47.0112/42.8986/44.7678s.
    Thus the parent slowed by16.9437%,
    while candidate/current-parent total wall ratio is1.008337223634.
    Automatic approval review rejected execution of submit_v2.py before creating
    Resources:29 retired groups including one preserved scorer failure;
    20 new native and6 client full calls,10573027832840F total;
    guarded wall sum977.328337s,peakRSS6657081344B,zero added swap.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_lower_moment_panel_v1/candidate.py](code/full_lower_moment_panel_v1/candidate.py)
- [code/full_lower_moment_panel_v1/common.py](code/full_lower_moment_panel_v1/common.py)
- [code/full_lower_moment_panel_v1/confirmation.py](code/full_lower_moment_panel_v1/confirmation.py)
- [code/full_lower_moment_panel_v1/finalize.py](code/full_lower_moment_panel_v1/finalize.py)
- [code/full_lower_moment_panel_v1/panel.py](code/full_lower_moment_panel_v1/panel.py)
- [code/full_lower_moment_panel_v1/poll_submission.py](code/full_lower_moment_panel_v1/poll_submission.py)
- [code/full_lower_moment_panel_v1/prepare.py](code/full_lower_moment_panel_v1/prepare.py)
- [code/full_lower_moment_panel_v1/prepare_package.py](code/full_lower_moment_panel_v1/prepare_package.py)
- [code/full_lower_moment_panel_v1/prepare_package_v2.py](code/full_lower_moment_panel_v1/prepare_package_v2.py)
- [code/full_lower_moment_panel_v1/qualify_boundary.py](code/full_lower_moment_panel_v1/qualify_boundary.py)
- [code/full_lower_moment_panel_v1/qualify_matched_runtime.py](code/full_lower_moment_panel_v1/qualify_matched_runtime.py)
- [code/full_lower_moment_panel_v1/record_submission_state_v1.py](code/full_lower_moment_panel_v1/record_submission_state_v1.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/full_lower_moment_panel_v1/score.py](code/full_lower_moment_panel_v1/score.py)
- [code/full_lower_moment_panel_v1/score_confirmation.py](code/full_lower_moment_panel_v1/score_confirmation.py)
- [code/full_lower_moment_panel_v1/score_v2.py](code/full_lower_moment_panel_v1/score_v2.py)
- [code/full_lower_moment_panel_v1/submit.py](code/full_lower_moment_panel_v1/submit.py)
- [code/full_lower_moment_panel_v1/submit_v2.py](code/full_lower_moment_panel_v1/submit_v2.py)
- [code/full_lower_moment_support_v1/candidate.py](code/full_lower_moment_support_v1/candidate.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [REEXAMINATION_INDEX_20260919.md#L880](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L880)
- [REEXAMINATION_INDEX_20260919.md#L882](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L882)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [RESULT_v2.md](RESULT_v2.md) · result
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-full-lower-moment-support-passes-broader-panel-adjusted-ratio) · [Copy provenance](../../../PROVENANCE.json)

