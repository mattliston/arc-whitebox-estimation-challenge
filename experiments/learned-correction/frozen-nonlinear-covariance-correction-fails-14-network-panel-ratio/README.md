# Frozen nonlinear covariance correction fails 14-network panel: ratio 1.009582713004, 5/14 wins

2026-09-21 · experiment · rejected · measured

The unchanged payload that passed the two-network development check was evaluated on a fixed broader panel.

Recorded measurement:

    The native development ratio was .9935549509494914. The identical final528-parameter payload was carried into the predeclared panel without fitting or tuning. The old checkpoint81 proxy gates remain failed. These networks were excluded from that fit but have prior research use; they are not pristine holdouts or official evaluation.
    Panel: pooled adjusted ratio **1.009582713004**, 5/14 wins, worst 1.039047149875; passed=False.
    |40|1.025225682858|1.000000000000|1.025225682858|
    |41|1.039047149875|1.000000000000|1.039047149875|
    |43|0.986637143752|1.000000000000|0.986637143752|
    |44|0.995031773361|1.000000000000|0.995031773361|
    |45|0.995751985354|1.000000000000|0.995751985354|
    |46|1.019171349383|1.000000000000|1.019171349383|
    |47|1.001547119163|1.000000000000|1.001547119163|
    |49|1.036126912772|1.000000000000|1.036126912772|
    |50|1.019927659193|1.000000000000|1.019927659193|
    |51|1.033444264224|1.000000000000|1.033444264224|
    |52|0.986470703125|1.000000000000|0.986470703125|
    |53|0.969907761911|1.000000000000|0.969907761911|
    |54|1.015590574695|1.000000000000|1.015590574695|
    |55|1.004287274583|1.000000000000|1.004287274583|
    Panel gate: pooled<1, at least8/14 wins, worst<=1.10. Confirmation gate: pooled<1, at least5/8 wins, worst<=1.10. The complete roster predictions and each arm's own cold cost were sealed before ordinary1B targets. Pooled scores average actual adjusted losses. Every fresh call includes module/static setup. The first8 outputs match the bank exactly; all16 unfloored variance vectors are positive and every full source is finite and symmetric. Saved pair-local sources pass independent FP32/FP64 references. Numerical caches retire.
    16 guarded groups and 14 full native calls retired; controller48860 terminal. Total 5784125253704 F, peak sampled RSS6431129600 B, self-report6431440896 B, zero added swap. 98 source cells verified. Artifacts 2.272 MB before closeout. No new fit, sampling, download or GPU work.
    Official331684 remains adjusted3.5844144212297072e-9, raw1.949715441540434e-8, 100/100 complete and zero failures.4e-9 is banked;3/2/~1e-9 remain unmet. The overall goal remains active and qualified uploads remain authorized.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_edge_nonlinear_panel_v1/baseline_audit.py](code/current_edge_nonlinear_panel_v1/baseline_audit.py)
- [code/current_edge_nonlinear_panel_v1/common.py](code/current_edge_nonlinear_panel_v1/common.py)
- [code/current_edge_nonlinear_panel_v1/finalize.py](code/current_edge_nonlinear_panel_v1/finalize.py)
- [code/current_edge_nonlinear_panel_v1/native.py](code/current_edge_nonlinear_panel_v1/native.py)
- [code/current_edge_nonlinear_panel_v1/preflight.py](code/current_edge_nonlinear_panel_v1/preflight.py)
- [code/current_edge_nonlinear_panel_v1/prepare.py](code/current_edge_nonlinear_panel_v1/prepare.py)
- [code/current_edge_nonlinear_panel_v1/prepare_v2.py](code/current_edge_nonlinear_panel_v1/prepare_v2.py)
- [code/current_edge_nonlinear_panel_v1/qualify.py](code/current_edge_nonlinear_panel_v1/qualify.py)
- [code/current_edge_nonlinear_panel_v1/reference.py](code/current_edge_nonlinear_panel_v1/reference.py)
- [code/current_edge_nonlinear_panel_v1/run.py](code/current_edge_nonlinear_panel_v1/run.py)
- [code/current_edge_nonlinear_panel_v1/score.py](code/current_edge_nonlinear_panel_v1/score.py)
- [code/current_edge_nonlinear_native_v1/candidate.py](code/current_edge_nonlinear_native_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/current_edge_nonlinear_native_v1/common.py](code/current_edge_nonlinear_native_v1/common.py)
- [code/current_edge_nonlinear_native_v1/native.py](code/current_edge_nonlinear_native_v1/native.py)
- [code/current_edge_nonlinear_native_v1/reference.py](code/current_edge_nonlinear_native_v1/reference.py)
- [code/full_lower_covariance_edge_native_v1/candidate.py](code/full_lower_covariance_edge_native_v1/candidate.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)
- [REEXAMINATION_INDEX_20260919.md#L626](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L626)
- [REEXAMINATION_INDEX_20260919.md#L631](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L631)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-frozen-nonlinear-covariance-correction-fails-14-network-panel-ratio) · [Copy provenance](../../../PROVENANCE.json)

