# Batching covariance and residual-plane products cuts server CPU 2.12% and residual time 12.6 ms

2026-09-23T11:22Z · experiment · adopted · measured

Sharing kernel stages across five independent dense products preserved predictions. It added a small first-use binding cost but reduced requests and runtime.

Recorded measurement:

    - 2026-09-23 04:22 [engineer] PRODUCT BATCHING (--batch, batch_fragment.py, make_batch_layout.py): each layer's five FLMS 1024^3 D5 products (covariance + star) run as two P-batched kernel calls (stage 1: C32@w, w^T@A32; stage 2: w^T@first, pw@w, (w^T*w^T)@pushed^T); level 1 per product, levels 2-5 + leaf + unwinds 5-2 shared; batched layouts (P=2: 142.6M, P=3: 213.9M floats) inside the existing arena, leaf disjoint so copy elimination always applies. File fable_engineer_v1/candS5e_batchv1_mem.py (cf49f47f) = S5e recipe + --batch + diet. Vs fired S5e: predictions BIT-IDENTICAL r35/r100/r37/r39; util identical on warm nets, cold net +0.38% (one-time batched frame binding, ~+0.02%/MLP per 20-net worker); numpy calls r35 37,172 -> 34,204, r100 34,990 -> 31,882 (~-8.5%, ~-1.5 s platform, ~-12 ms residual); cached arrays +564 constant. Sent to engineer2's paired gate (CPU, residual, handles, live peak vs S5e).
    2026-09-23 04:44 [engineer2] S6b GATE PASS (porter's stack_S6b_mem.py a514c21b = S5e recipe + refit epoch2 + --batch + mem,yright,arena; run concurrently with the fired S5e over the same 20 distinct nets, paired20.py g20S6b g20S5eC). Predictions bit-exact vs stack_S6_mem.py (r35/r100, 16 layers). Server CPU -2.12% mean / -2.38% median (-5.6..+2.5%); residual -12.6 ms mean, COLD net 1 (r35) 217 vs 230 ms (-13), max 217 ms; round trips -2.6k on the cold net, -3.2k (-7%) on warm nets; util -0.055% mean (cold r35 +0.34% one-time batched-frame binding, warm -0.075% each); handles +569 constant (74,701 vs 74,132 after 20); live peak 2,348-2,383 MB (S5e 2,386-2,404). Refit MSE r35 -0.38%, r100 -0.36%. Platform projection from S5e: cold-MLP residual ~0.32 s, wall ~-1.8 s mean.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/batch_fragment.py](code/fable_engineer_v1/batch_fragment.py)
- [code/fable_engineer_v1/make_batch_layout.py](code/fable_engineer_v1/make_batch_layout.py)
- [code/fable_engineer_v1/candS5e_batchv1_mem.py](code/fable_engineer_v1/candS5e_batchv1_mem.py)
- [code/fable_porter_v1/stack/stack_S6b_mem.py](code/fable_porter_v1/stack/stack_S6b_mem.py)
- [code/fable_engineer2_v1/paired20.py](code/fable_engineer2_v1/paired20.py)
- [code/fable_porter_v1/stack/stack_S6_mem.py](code/fable_porter_v1/stack/stack_S6_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L188](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L188)
- [TEAM_LOG_2026-09-22_to_09-26.md#L196](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L196)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332005: joint closure refit, batched products, and memory-layout changes](../../../submissions/332005/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-batching-covariance-residual-plane-products-cuts-server-cpu-2) · [Copy provenance](../../../PROVENANCE.json)

