# Shared left transforms and repeated-square reuse save 0.34% work with identical predictions

2026-09-24T18:41Z · experiment · adopted · measured

A per-column package reuses the common left transform across packets, repeated squares and compact gated columns. It reduces requests and composes exactly with the dead-side row cut.

Recorded measurement:

    - 2026-09-24 11:41 [engineer] --colpack READY (fable_engineer_v1/RESULT.md E-49): candS11cp_mem.py c6ca3ebf = S11 acf67b9d + the exact per-column package (shared left transform across six-level packets via a 138 MB side buffer; squares reused in the read; compact gated columns). Preds BIT-IDENTICAL r35/r100; bill -0.293% cold / -0.345% warm; einsum gate PASS; validate PASS; ABBA server check: requests -508/-689, round trips -453/-634, server CPU -10.0/-4.7 s (noisy, no increase), residual -0.10/-0.07 s, live-array peak +150 MB, process peak RSS +28 MB. Build: post-diet 'python3 fable_engineer_v1/build_colpack.py <S11_mem.py> <out.py>' (after build_e2mem.py; cannot run inside build_stack.sh because the diet rewrites the edited loop lines). Handed to the porter.
    - 2026-09-24 12:00 lead: --colpack (engineer E-49, candS11cp_mem.py c6ca3ebf) bit-identical, bill -0.29% cold / -0.35% warm, requests -500..-700/net; a post-diet stage, now with the porter. Six-level reads CLOSED (engineer2: 70/117 read classes need 64-padding, so all reads +0.39% and the best 28 classes only -0.25% for +3.5-4.2 s). Wall-room candidates: copy-free Winograd levels (engineer) and less frame binding / column keying E-44 (engineer2), both re-priced against S11 + deadcut35 (+R320) room (engineer2: S12 pessimistic room ~8.8 s, additive, pending the measured pair).
    2026-09-24 11:48 porter: colpack verified (my build = engineer's c6ca3ebf; predictions identical to S11 on 6 nets, flops -0.34% pooled; exact under deadcut too; S11+colpack+deadcut flops -0.53..-0.89% vs S11). Chain order: stack [--rank] -> diet -> colpack -> deadcut. S12-full (R320+colpack+deadcut) CPU pair staged after the r320 watcher.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/candS11cp_mem.py](code/fable_engineer_v1/candS11cp_mem.py)
- [code/fable_engineer_v1/build_colpack.py](code/fable_engineer_v1/build_colpack.py)
- [code/fable_engineer2_v1/build_e2mem.py](code/fable_engineer2_v1/build_e2mem.py)
- [code/fable_porter_v1/build_stack.sh](code/fable_porter_v1/build_stack.sh)
- [code/fable_engineer2_v1/hold2.py](code/fable_engineer2_v1/hold2.py)
- [code/fable_engineer_v1/colpack_fragment_e1.py](code/fable_engineer_v1/colpack_fragment_e1.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L357](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L357)
- [TEAM_LOG_2026-09-22_to_09-26.md#L364](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L364)
- [TEAM_LOG_2026-09-22_to_09-26.md#L365](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L365)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

## Related work

- shipped in: [Submission #332195: column packing, inactive-row selection, six-level reads, and in-place tree products](../../../submissions/332195/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-shared-left-transforms-repeated-square-reuse-save-0-34) · [Copy provenance](../../../PROVENANCE.json)

