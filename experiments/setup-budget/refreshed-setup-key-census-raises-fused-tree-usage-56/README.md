# Refreshed setup-key census raises fused-tree usage from 56.7% to approximately 75%

2026-10-01T23:27Z · experiment · adopted · measured

The prebind list (the keys bound in setup, the first 128 of them fused) was fitted to old census networks: on 10 fresh networks only 377 of its 649 keys were ever asked for and 60.5% of the 914 tree uses per network fell on the first 128. A list regenerated from a fresh census (most-used 128 keys first, then by expected saved bind flops) puts 74.7-75.1% of uses on the fused keys out of sample, is byte-identical by construction, saves 0.05-0.07% of bill and 5-10 Gflop of setup; mainly a request lever (about 180 more fused tree uses per network). It entered S36.

Recorded measurement:

    in sample 76.5% of uses on first 128, lazy binds 0.32% with 480 keys bound; out of sample: first 128 keys 56.7% -> 74.7-75.1%, first 480 keys 74.3% -> 89.6%, lazy-bind flops with 480 keys 0.510 -> 0.438% and 0.465 -> 0.418% of bill; setup 24-30 Gflop instead of 35; holdout: predictions byte-identical 10 of 10, charged ops -2.2%, bill -0.006%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a13_prebind`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/regen_prebind.py](code/regen_prebind.py)
- [code/prebind_hfull_v1.py](code/prebind_hfull_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L57](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L57)
- [TEAM_LOG_2026-10-01_to_10-22.md#L61](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L61)
- [TEAM_LOG_2026-10-01_to_10-22.md#L66](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L66)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333429: float32 arithmetic, half-zero strips, refreshed setup keys and separate tree levels](../../../submissions/333429/README.md)

[Research graph](../../../graph/index.html#node=2026-10-01-refreshed-setup-key-census-raises-fused-tree-usage-56) · [Copy provenance](../../../PROVENANCE.json)

