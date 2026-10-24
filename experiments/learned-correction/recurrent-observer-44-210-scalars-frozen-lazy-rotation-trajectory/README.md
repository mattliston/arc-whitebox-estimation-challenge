# Recurrent observer (44,210 scalars) on the frozen lazy-rotation trajectory: TRAIN20 adjusted ratio 1.00353 vs <=0.95

2026-09-20 · experiment · rejected · measured

An independent recurrent correction warm-started from a copy of the current learned heads P was trained once (30 epochs, 150 AdamW updates, 600 backwards, 44,210 trainable scalars) on the frozen physical trajectory of the accepted lazy-rotation card, replacing only the final neural correction. Its endpoint misses the fixed gate even with zero added cost (ratio 0.98376480255369314), so no native port or CHECK scoring followed.

Recorded measurement:

    TRAIN20 adjusted 3.92095885326582e-9 vs parent 3.9071715315547946e-9, ratio 1.003528721889909 (gate <=0.95), 7/20 adjusted wins; zero-added-cost ratio 0.98376480255369314, raw pooled 0.98372547653513487 (18/20 raw wins); replay 4552455498 F per call; 26 parent calls 10354905432271 F

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lazy_frozen_trajectory_observer_v1/capture_all.py](code/lazy_frozen_trajectory_observer_v1/capture_all.py)
- [code/lazy_frozen_trajectory_observer_v1/prepare_fit_v2.py](code/lazy_frozen_trajectory_observer_v1/prepare_fit_v2.py)
- [code/lazy_frozen_trajectory_observer_v1/prepare_inputs.py](code/lazy_frozen_trajectory_observer_v1/prepare_inputs.py)
- [code/lazy_frozen_trajectory_observer_v1/fit.py](code/lazy_frozen_trajectory_observer_v1/fit.py)
- [code/lazy_frozen_trajectory_observer_v1/qualify_neural.py](code/lazy_frozen_trajectory_observer_v1/qualify_neural.py)
- [code/lazy_frozen_trajectory_observer_v1/replay_fnp.py](code/lazy_frozen_trajectory_observer_v1/replay_fnp.py)
- [code/lazy_frozen_trajectory_observer_v1/capture.py](code/lazy_frozen_trajectory_observer_v1/capture.py)
- [code/lazy_frozen_trajectory_observer_v1/common.py](code/lazy_frozen_trajectory_observer_v1/common.py)
- [code/lazy_frozen_trajectory_observer_v1/model.py](code/lazy_frozen_trajectory_observer_v1/model.py)
- [code/lazy_frozen_trajectory_observer_v1/prepare_fit.py](code/lazy_frozen_trajectory_observer_v1/prepare_fit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L44](NEGATIVE_RESULT.md#L44)
- [NEGATIVE_RESULT.md#L58](NEGATIVE_RESULT.md#L58)
- [NEGATIVE_RESULT.md#L59](NEGATIVE_RESULT.md#L59)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-recurrent-observer-44-210-scalars-frozen-lazy-rotation-trajectory) · [Copy provenance](../../../PROVENANCE.json)

