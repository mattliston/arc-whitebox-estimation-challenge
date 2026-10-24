# Final recurrent observer on Gaussian propagation reaches training MSE 1.0376e-6, above the 2.5e-8 limit

2026-09-20 · experiment · rejected · measured

A cheap pure Gaussian (order-6) physical trajectory with no third or fourth carrier and no learned feedback was given the current 64-state/48-message weight-graph observer trained final-only for 60 epochs (300 updates); it removes 72.23% of the Gaussian host's TRAIN20 error, but the remaining 1.037641433545e-6 is 41.51 times the advancement limit and, even at the 0.1 multiplier floor, 26.56 times the lazy-rotation card's adjusted error on the same targets, so the low-cost-host route closed without CHECK scoring.

Recorded measurement:

    TRAIN20 Gaussian6 host 3.736508635518e-6; fitted observer 1.037641433545e-6 (ratio 0.277703475293); limit 2.5e-8 missed by 41.51x; floor-adjusted 1.037641433545e-7 vs lazy 3.907171531555e-9, ratio 26.55735550, 0/20 wins; host plus features about 66.6492 GF, neural replay 4.552455498 GF

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/capture.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/capture.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/prepare.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/prepare.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/fit.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/fit.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/observed_forecast.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/observed_forecast.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/native_neural.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/native_neural.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/qualify_neural.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/qualify_neural.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/finalize.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/finalize.py)
- [code/astra_phase2_reexamination_v1/gaussian_terminal_skew_v2/estimator.py](code/astra_phase2_reexamination_v1/gaussian_terminal_skew_v2/estimator.py)
- [code/astra_phase2_reexamination_v1/gaussian_terminal_skew_v2/stable_gaussian.py](code/astra_phase2_reexamination_v1/gaussian_terminal_skew_v2/stable_gaussian.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/common.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/common.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/features.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/features.py)
- [code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/model.py](code/codex_resume_20260919_v1/gaussian_frozen_observer_v1/model.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L15](RESULT.md#L15)
- [RESULT.md#L16](RESULT.md#L16)
- [RESULT.md#L17](RESULT.md#L17)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-final-recurrent-observer-gaussian-propagation-reaches-training-mse-1) · [Copy provenance](../../../PROVENANCE.json)

