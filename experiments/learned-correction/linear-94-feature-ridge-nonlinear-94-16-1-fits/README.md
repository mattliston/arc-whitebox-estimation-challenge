# Linear (94-feature ridge) and nonlinear (94-16-1) fits of the terminal variance correction fail held-out networks

2026-10-15T03:57:19Z · experiment · rejected · measured

Fitting the terminal variance correction from the existing 94 per-unit features by ridge regression on three captured networks did not generalise to an independent fourth; adding 64 propagated-state features made it worse, and a 94-16-1 SiLU head trained with Adam was far worse under leave-one-network-out. The oracle headroom exists but the available linear features do not predict it.

Recorded measurement:

    ridge lambda 0.1 within TRAIN16 blocked held-out MSE -0.201974% (target R2 0.0145); fit 16/259/277 then independent 287: MSE +0.214572% worse (R2 0.0459); propagated-state features: baseline 94 +1.794% geometric, 94+64 +3.850%, incremental worse on all 3 (+2.019%); SiLU head 500-step Adam: LONO pooled MSE +41.989790% (folds +93.573645/+24.107307/+12.637921%), positive-scale ceiling 0.154171%; terminal hook cost 389,120 FLOPs / 6 ops

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_native_terminal_linear_refit_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fit_network_block_ridge_HELD_v1.py](code/fit_network_block_ridge_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3242](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3242)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3244](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3244)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3245](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3245)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3246](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3246)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3251](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3251)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3311](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3311)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3313](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3313)

<details>
<summary>All result files and records (5)</summary>

- [NETWORK_BLOCK_CV_RESULT_v1.json](NETWORK_BLOCK_CV_RESULT_v1.json) · result
- [PROPAGATED_STATE_FIXED_SCREEN_v1.json](PROPAGATED_STATE_FIXED_SCREEN_v1.json) · result
- [BLOCKED_CV_RESULT_v1.json](BLOCKED_CV_RESULT_v1.json) · result
- [ALL24_DECISION_astra_v1.json](ALL24_DECISION_astra_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Terminal variance oracle: true layer-14 post-variance -9.2% final MSE; diagonal part ~0, covariance/transport -10.3%](../../../records/research/2026-10-15-terminal-variance-oracle-true-layer-14-post-variance-9/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-linear-94-feature-ridge-nonlinear-94-16-1-fits) · [Copy provenance](../../../PROVENANCE.json)

