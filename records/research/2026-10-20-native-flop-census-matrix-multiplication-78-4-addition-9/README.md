# Native FLOP census: matrix multiplication 78.4%, addition 9.4%, subtraction 6.7%

2026-10-20T05:13:12Z · finding · informational · measured

A warm native prediction on network 35 cost 236.9 GFLOP and 38,060 operations. Float32 accounted for 99.4% and float64 0.5%, leaving little precision-conversion opportunity.

Recorded measurement:

    - [fable-lead] 2026-10-20T05:13:12Z BILL ANATOMY (flopscope op_log of one warm in-process predict, net 35, 236.9 GFLOP = util 0.1077 in-process): float32 99.4% / float64 0.5% (no precision lever); matmul 78.4%, add 9.4% + subtract 6.7% (the Winograd/Smirnov blocked kernels' pre/post-additions, shapes (7,7,7,...)), multiply 1.8%, qr 0.6%; 38,060 ops per net (add 10.9k, subtract 8.3k, multiply 5.0k, matmul 1,650). Nothing cheap left on the bill side; flops_by_dtype_v1.py.

## Programs

- [flops_by_dtype_v1.py](flops_by_dtype_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6136](../../TEAM_LOG_2026-10-01_to_10-22.md#L6136)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-native-flop-census-matrix-multiplication-78-4-addition-9) · [Copy provenance](../../../PROVENANCE.json)

