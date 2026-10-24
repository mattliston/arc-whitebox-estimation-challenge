# Depth-4 arena runtime control on network 35: official client-server drift measured

2026-08-27 · experiment · informational · measured

The unchanged scored depth-4 third-cumulant arena (D4) bank also returns through the official client/server path with generic native/remote drift: RMS 2.67e-9 and maximum 1.1920928955078125e-7. Byte drift is thus not depth-5 third-cumulant arena (D5)-specific and does not explain the board failures.

Recorded measurement:

    The unchanged scored D4 bank also returns through the official client/server path with generic native/remote drift: RMS 2.67e-9 and maximum 1.1920928955078125e-7. Byte drift is thus not D5-specific and does not explain the board failures.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d23_d4_remote_rank35_control_v1/compare_remote_native.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d23_d4_remote_rank35_control_v1/compare_remote_native.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d23_d4_remote_rank35_control_v1/run_instrumented_server.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d23_d4_remote_rank35_control_v1/run_instrumented_server.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d23_d4_remote_rank35_control_v1/run_remote_rank35_client.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d23_d4_remote_rank35_control_v1/run_remote_rank35_client.py)

## Evidence

- [D23_D4_REMOTE_RANK35_CONTROL_RESULT.json#L1](D23_D4_REMOTE_RANK35_CONTROL_RESULT.json#L1)
- [D23_SERVER_TELEMETRY.json#L1](D23_SERVER_TELEMETRY.json#L1)
- [D23_D4_REMOTE_NATIVE_DRIFT.json#L1](D23_D4_REMOTE_NATIVE_DRIFT.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [D23_D4_REMOTE_RANK35_CONTROL_RESULT.json](D23_D4_REMOTE_RANK35_CONTROL_RESULT.json) · result
- [D23_SERVER_TELEMETRY.json](D23_SERVER_TELEMETRY.json) · result
- [D23_D4_REMOTE_NATIVE_DRIFT.json](D23_D4_REMOTE_NATIVE_DRIFT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-depth-4-arena-runtime-control-network-35-official-client) · [Copy provenance](../../../PROVENANCE.json)

