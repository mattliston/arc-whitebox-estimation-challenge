# Lazy-bind v4 contiguous-leaf boxes: -0.08% bill for +1-5% server instructions and +0.1-4.6% client residual, dropped

2026-10-16T16:44:50Z · experiment · rejected · measured

lazybind v4 made multi-aspect leaf boxes on demand (one box per leaf width, leaves C-contiguous) to remove the remaining lazy tree binds of _ipt_view. The C family alone cost +1.04% process instructions for -0.015% flops and was dropped; the A/B menus gave byte-identical predictions on all 8 session networks for about -0.08% flops but +1.1-5.0% server instructions and up to +4.6% client residual instructions, a rate of about 0.04% bill per server-second, below the 0.1%/s bar the Smirnov plans were held to. The lead dropped it for the zero-failure build.

Recorded measurement:

    Native warm pair: v4 full (A/B/C) -0.122% flops / +3.30% process instructions; C alone -0.015% / +1.04%. 8-net sessions: v4ab flops nets 2-8 mean -0.083%, requests -0..-20, server instructions +1.8..+5.0%; v4b -0.076%, server +1.1..+3.3%, client residual instructions +0.1..+4.6% (rank 100 +0.85% / +1.14%).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_lazybind4_v1/build_lb4_v1.py](code/opus_lazybind4_v1/build_lb4_v1.py)
- [code/opus_lazybind4_v1/cand_s3_lb4ab_v1.py](code/opus_lazybind4_v1/cand_s3_lb4ab_v1.py)
- [code/opus_lazybind4_v1/cand_s3_lb4_v1.py](code/opus_lazybind4_v1/cand_s3_lb4_v1.py)
- [code/opus_passrel_v1/stack_v1/cand_s3.py](code/opus_passrel_v1/stack_v1/cand_s3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4326](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4326)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4352](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4352)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4355](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4355)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4356](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4356)

<details>
<summary>All result files and records (3)</summary>

- [fragment_lazybind_v4.py.txt](fragment_lazybind_v4.py.txt) · code
- [fragment_lazybind_v4b.py.txt](fragment_lazybind_v4b.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Billed-view census: basic indexing free, reshape bills its input size (0.69% warm), lazy tree binds 0.29% warm](../../../records/research/2026-10-16-billed-view-census-basic-indexing-free-reshape-bills-input/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-lazy-bind-v4-contiguous-leaf-boxes-0-08-bill) · [Copy provenance](../../../PROVENANCE.json)

