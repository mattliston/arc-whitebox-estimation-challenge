# Saved-state final-head variants using reflection, Hermite features and nonlinear models all stay within ±0.08%

2026-10-17T04:05:37Z · experiment · rejected · measured

Saved-only fits of cheap corrections to the final state on the captured TRAIN32/TRAIN97 trajectories were all closed with no deployable gain: a reflected learned-head fixed half (+0.031675%), a label-free Hermite calibration (-0.0086%), a full-response GOP2 M2 metric (+0.007096%), a Jensen clamp (+0.018086%), a pooled 4-PCA context head (-0.012976%), a PCA16 whitened squared-state readout (+0.025018%), a 64-16-1 SiLU residual MLP (+0.078%), and a histogram-boosted oblivious tree head (-0.015463% against a -0.4% gate). The held 49 truths were never opened.

Recorded measurement:

    Reflected head TRAIN32 +0.031675% SE .044129 (16/32); Hermite TRAIN97 blocked 5-fold -.00862756% SE .01565195 (pooled -.00677148, 57/97); GOP2 M2 +0.007096% SE .003014 (10/32); Jensen +.018086% SE .043828 (17 wins); pooled context 4 -.012976% SE .137403 (51 wins); state quadratic +.025018% SE .021298% (46/97); residual MLP +.0783715% SE .138784 (44/97); oblivious 32 trees -.015463% (log SE .049839), 51/97.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_crunch_state_mlp_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fit_v1.py](code/fit_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4758](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4758)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4764](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4764)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4768](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4768)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4770](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4770)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4801](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4801)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4807](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4807)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4873](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4873)

<details>
<summary>All result files and records (8)</summary>

- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [CLOSURE_v1-9880d082.json](CLOSURE_v1-9880d082.json) · record
- [CLOSURE_v1-0a97e65e.json](CLOSURE_v1-0a97e65e.json) · record
- [CLOSURE_v1-15a94e2c.json](CLOSURE_v1-15a94e2c.json) · record
- [CLOSURE_v1-fb2e3795.json](CLOSURE_v1-fb2e3795.json) · record
- [CLOSURE_v1-b56f4f53.json](CLOSURE_v1-b56f4f53.json) · record
- [CLOSURE_v1-6a291641.json](CLOSURE_v1-6a291641.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-saved-state-final-head-variants-using-reflection-hermite-features) · [Copy provenance](../../../PROVENANCE.json)

