# Client residual is a fixed per-call cost: ~42K instructions per request, flat across call forms; only fewer calls help

2026-10-16T17:58:42Z · finding · informational · measured

A cProfile of LR2 on the boundary showed the candidate own Python is small (0.135 s self) and the residual is dominated by the flopscope per-call span machinery, so residual is about the number of fnp calls (requests and free view or shape calls) times a fixed edge cost. A client-server micro-benchmark of 2,000 calls per form then found every public call form within +-3% (operator vs function, out=, scalar vs 0-d array, einsum, matmul, getitem, reshape, transpose), so no cheaper call form exists; multi-output views cost about 4.4K per extra handle; flopscope 0.12.0 has no batching API.

Recorded measurement:

    Per-call residual instructions: a*b 42.2K, fnp.multiply 42.1K, out= 42.5K, a+b 42.5K / fnp.add 42.1K, einsum 42.5K, matmul 42.5K / 42.2K, getitem 41.9-42.0K, reshape 42.1K / 43.4K, .T 42.3K, sum 42.3K; span machinery alone 18.4K instructions / 0.92 us per call; candidate own Python ~2-5K per request (measured 45.5K per request on the warm net). LR2 profile: _globals_are_internal / _caller_is_internal 122,738 calls, _counted_span + contextlib 164,026, wrapped 41,288 per warm network.

## Programs

- [mb_client_v1.py](mb_client_v1.py)
- [mb_run_v1.py](mb_run_v1.py)
- [cand_lr2_copy_v1.py](cand_lr2_copy_v1.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4284](../../TEAM_LOG_2026-10-01_to_10-22.md#L4284)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4429](../../TEAM_LOG_2026-10-01_to_10-22.md#L4429)

<details>
<summary>All result files and records (2)</summary>

- [mb_v1.json](mb_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-client-residual-fixed-per-call-cost-42k-instructions-per) · [Copy provenance](../../../PROVENANCE.json)

