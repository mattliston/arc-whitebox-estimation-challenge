# Terminal 64-state ridge readout refit: leave-one-network-out MSE +0.06..+0.33%

2026-10-20T04:25:12Z · experiment · rejected · measured

A 32-network refit of the terminal readout improved in-sample MSE by -0.14..-0.43% but worsened leave-one-network-out MSE. The candidate builder was left unused.

Recorded measurement:

    - [fable-lead] 2026-10-20T04:25:12Z TERMINAL READOUT REFIT CLOSED: gnn_step l=15 readout (64-dim state, exact basis, reconstruction 1e-10) refit by ridge on 32 nets (28 fresh + 35/100/37/39): leave-one-net-out +0.06..+0.33% (worse), in-sample -0.14..-0.43%, head-off +4.6%. The current head is at this family's optimum; the final residual is not predictable from the GNN state (matches 10-10 'L15 heads CLOSED'). Tooling: native_head_probe_v1.py, fit_head15_v1.py, build_head15_candidate_v1.py (unused). Next: 13 queued arms (pe34/gnn gain/k415/dcv floor) in the lanes + a 6-net paired native pre-screen (native_screen_v2.py, prescreen13_*.log).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/fable_oct19_kc_v1/native_head_probe_v1.py](code/lead/runs/fable_oct19_kc_v1/native_head_probe_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/fit_head15_v1.py](code/lead/runs/fable_oct19_kc_v1/fit_head15_v1.py)
- [code/lead/runs/fable_oct19_kc_v1/build_head15_candidate_v1.py](code/lead/runs/fable_oct19_kc_v1/build_head15_candidate_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6132](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6132)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-terminal-64-state-ridge-readout-refit-leave-one-network) · [Copy provenance](../../../PROVENANCE.json)

