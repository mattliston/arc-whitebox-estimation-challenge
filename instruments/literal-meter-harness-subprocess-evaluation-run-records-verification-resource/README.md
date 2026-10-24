# Literal meter harness with subprocess evaluation, run records, verification and resource gates

2026-08-23T17:59Z · tool · adopted · derived

Every Phase 2 estimator was metered through one driver (run_synthetic_meter.py) that runs the estimator under the official whestbench 0.16.0 subprocess harness with the flopscope 0.12.0 client and server on a shared synthetic fixture (one 1024x16 network, 1000 synthetic truth samples, seed 20260823, 4 threads), writes a receipt with the metered FLOPs, wall, backend, overhead and residual times and the sampled process-tree peak RSS, and checks it with verify_meter_run.py against the 2^41 budget, the 10x score-floor predicate, the 0.4 s residual, the 120 s wall and the 8 GiB memory limit. Every later resource claim in the slice cites one of these receipts.

Recorded measurement:

    ladder summary: hermite_order10 103,923,320,832 FLOPs, wall 1.1416656249202788 s, residual 0.10807238332927227 s, RSS 2,195,210,240 bytes; cp3_cap512 136,462,170,114 FLOPs; score-floor predicate 10 x flops_used <= 2^41 (max integer 219,902,325,555)

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_extended_wall_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_extended_wall_diagnostic.py)
- [code/estimator.py](code/estimator.py)

## Evidence

- [METER_LADDER_SUMMARY.json#L1](METER_LADDER_SUMMARY.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [source_README.md](source_README.md) · record
- [METER_LADDER_SUMMARY.json](METER_LADDER_SUMMARY.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-08-23-literal-meter-harness-subprocess-evaluation-run-records-verification-resource) · [Copy provenance](../../PROVENANCE.json)

