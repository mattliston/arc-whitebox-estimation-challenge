# Layer-3 teacher-attribution runner stopped before data access when process supervision was denied

2026-08-23T19:10Z · incident · failed · measured

The audited V7 design for attributing the layer-3 error to source versus consumer (teacher substitutions of mean, variance, K3 and K4 on the mini split) repaired the two custody blockers of V6 and passed source review, but its sole one-use invocation died at the first whole-process-group RSS sample because the sandbox returned PermissionError on /bin/ps, and the process-group containment call failed the same way. The invocation tombstone was consumed with no data claim, no decoded target and no result; the independent audit confirmed no surviving worker. Later direct public-role diagnostics replaced the need to retry this form.

Recorded measurement:

    invocations_attempted 1, launcher_exit_code 1, physical_dataset_rows_decoded 0, target_values_decoded 0, results_present 0

## Programs

- [run_pilot.py](run_pilot.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../experiments/third-order-planes/third-cumulant-carrier-caps-5121792-small-endpoint-gains-layer/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1792_k3_readout/estimator.py](../../../experiments/third-order-planes/third-cumulant-carrier-caps-5121792-small-endpoint-gains-layer/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/cp3_cap1792_k3_readout/estimator.py)
- [../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/code/target_blind_mini.py](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/code/target_blind_mini.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/target_blind_mini_v2.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/target_blind_mini_v2.py)
- [../../../experiments/series-expansions/hermite-pair-series-order-sweep-order-4-reached-3/code/exact_layer_foothold/verify_exact_layer_foothold.py](../../../experiments/series-expansions/hermite-pair-series-order-sweep-order-4-reached-3/code/exact_layer_foothold/verify_exact_layer_foothold.py)
- [build_probe_asset.py](build_probe_asset.py)
- [build_probe_asset_v1law.py](build_probe_asset_v1law.py)
- [pilot_core.py](pilot_core.py)
- [launch_once.py](launch_once.py)
- [runtime_contract.py](runtime_contract.py)
- [science_core.py](science_core.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [worker.py](worker.py)
- [verify_pilot_static.py](verify_pilot_static.py)

</details>

## Evidence

- [EXECUTION_FAILURE_BOUNDARY_V7.json#L1](EXECUTION_FAILURE_BOUNDARY_V7.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [EXECUTION_FAILURE_BOUNDARY_V7.json](EXECUTION_FAILURE_BOUNDARY_V7.json) · result
- [INDEPENDENT_POST_FAILURE_AUDIT_V7.json](INDEPENDENT_POST_FAILURE_AUDIT_V7.json) · result
- [DESIGN.md](DESIGN.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-23-layer-3-teacher-attribution-runner-stopped-before-data-access) · [Copy provenance](../../../PROVENANCE.json)

