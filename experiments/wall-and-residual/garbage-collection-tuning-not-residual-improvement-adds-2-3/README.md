# Garbage-collection tuning is not a residual improvement and adds 2-3 GB peak resident memory

2026-09-24T23:12Z · experiment · rejected · measured

A more precise profile found collection inside dispatch spans already excluded from residual accounting. The collection policy was removed, leaving view and cache-entry reductions as the valid optimization.

Recorded measurement:

    - 2026-09-24 16:12 [engineer] RESIDUAL RANKED on S15 (#332207; residual_probe_v3/v4, client/server, every candidate function wrapped, requests counted, wrapper cost subtracted, GC and handle finalizers timed; wrapped total 263 of 266 ms cold). Per net cold/warm: (1) cyclic GC 42.5/48.4 ms (265/209 young passes, 2 full); harness-applied gc.set_threshold(20000,10,10) + gc.freeze() after each net -> 17.2/12.9 ms (-25/-36 ms), candidate untouched; (2) tree-kernel per-call client work 141 ms (~22k requests x ~5 us + proxy deaths; 35.5k handle finalizers/net); natural-order per-call views = -7 requests/-9 proxies per product, ~-12 ms; (3) physics loop + helpers ~60 ms (owners' diet); (4) tree binding 22/13 ms, merge ab views into the tree entry ~-1..-2 ms. My lane total ~-39 ms cold / -50 ms warm. Awaiting the lead's GO (item 1 is a process-wide GC policy).
    - 2026-09-24 16:22 lead: RESIDUAL TRIM PLAN (engineer, residual_probe_v3/v4 on S15; the wrapped sum is 263 of 266 ms cold). Ranked client-side costs:
      (1) Python cyclic GC 42/48 ms (full passes rescan cached view proxies): gc.set_threshold(20000,10,10) + gc.freeze() after each prediction gives -25/-36 ms.
      (2) Tree-kernel per-call views ~141 ms cold: natural-axis views give ~-12 ms, bit-exact, and touch engineer2's alt-basis code.
      (3) Physics loop ~60 ms, owners only.
      (4) Tree binding: merged cache entry ~-1..-2 ms.
      GO on one bit-exact flag on S16 with (1)+(2)+(4), expected ~-40..-50 ms cold local.
      MUST-PASS gates: bit-identical preds and bill; run_mem shows NO rise in max handles, live peak or RSS (delayed GC must not strand server proxies toward the 100k handle cap); paired residual gate; hasattr guard on gc.freeze.
    - 2026-09-24 16:21 [engineer] S16r BUILT (residual trim, lead GO 16:22): fable_engineer_v1/candS16r_mem.py c87a3dda = fired S16 (a4a1e0f7) + cat resid_fragment.py: GC threshold 20000 + guarded gc.freeze after import and each prediction (literal _RS_FREEZE), natural-order per-call views in the alt-basis tree kernel (engineer2 agreed), alt-basis basis views folded into the tree cache entry, the plain tree-side helpers made to raise (unreachable while _AB_ON). In process vs S16: byte-identical on r35/r100, bill +0, ops -1,455/-1,565. Einsum PASS, validate PASS. Queued under the lock: client/server identity, run_mem.sh S16 vs S16r over 4 nets (handles/live peak/RSS no-rise gate), residual_probe_v5 (cold/warm, plus per-pass cyclic-garbage counts to decide whether gc.freeze can strand proxies).
    - 2026-09-24 17:11 [engineer] RESIDUAL TRIM, GC PART IS WORTH ZERO: residual_probe_v6 labels each cyclic-GC pass by where it starts. On S16 and S16r2 (nets 35/100/37) EVERY pass starts inside a flopscope dispatch span, so GC time is dispatch, not residual. The client's cyclic garbage is flopscope's own per-request check closure (gc_garbage_probe; no array proxy ever in it, so no stranded handles). The threshold trims only 20-25 ms wall/net and adds 2-3 GB in-process peak RSS. View trim (the real cut): -1,455..-1,650 requests/net, bit-identical, bill identical; local residual vs S16 in four same-window comparisons -11.3/-7.2/-1.1/-11.4 ms (mean -7.8). S16r2 (candS16r2_mem.py 0a33c080, trim + GC threshold) run_mem ABBA 4 nets: max handles 23,344/23,424 vs S16 25,821/25,820; live peak and final live identical; server RSS +68/+4 MB, inside S16's own 143 MB spread. RECOMMENDED: S16r3b (candS16r3b_mem.py ca974768) = the view trim WITHOUT the GC threshold, gates queued (done ~17:45), then the porter's paired residual gate. Request fusion DEAD (fused B/C tree ops trip numpy's overlap solver -> output copies; A-only ~-5 ms steady). Next lever priced: setup() prebinding of recurring tree shapes, cold-net binding ~19 ms local; shape census running.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/candS16r_mem.py](code/fable_engineer_v1/candS16r_mem.py)
- [code/fable_engineer_v1/resid_fragment.py](code/fable_engineer_v1/resid_fragment.py)
- [code/fable_engineer2_v1/run_mem.sh](code/fable_engineer2_v1/run_mem.sh)
- [code/fable_engineer_v1/candS16r2_mem.py](code/fable_engineer_v1/candS16r2_mem.py)
- [code/fable_engineer_v1/candS16r3b_mem.py](code/fable_engineer_v1/candS16r3b_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L461](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L461)
- [TEAM_LOG_2026-09-22_to_09-26.md#L462](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L462)
- [TEAM_LOG_2026-09-22_to_09-26.md#L463](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L463)
- [TEAM_LOG_2026-09-22_to_09-26.md#L464](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L464)
- [TEAM_LOG_2026-09-22_to_09-26.md#L465](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L465)
- [TEAM_LOG_2026-09-22_to_09-26.md#L466](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L466)
- [TEAM_LOG_2026-09-22_to_09-26.md#L467](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L467)
- [TEAM_LOG_2026-09-22_to_09-26.md#L468](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L468)
- [TEAM_LOG_2026-09-22_to_09-26.md#L471](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L471)
- [TEAM_LOG_2026-09-22_to_09-26.md#L475](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L475)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-garbage-collection-tuning-not-residual-improvement-adds-2-3) · [Copy provenance](../../../PROVENANCE.json)

