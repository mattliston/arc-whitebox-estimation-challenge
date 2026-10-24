# Packetized six-level reads on eligible shapes save 0.363% work with rounding-level error

2026-09-24T19:02Z · experiment · adopted · measured

A sixth level inside existing read packets respects the arena size. Only divisible shapes that pass an exact work-versus-time rule are selected.

Recorded measurement:

      (2) --read6q (engineer2): in-situ six-level reads for 64-divisible shapes measured 2-3x better than priced: all divisible reads -0.70% r35 / -0.54% r100 for +3.1-3.6 s user CPU; best 8 per net -0.45..-0.70% for ~+3.2-3.9 s platform. Presets 'quarter' and 'half' are chosen after the measured S12 pair.
    2026-09-24 12:02 [engineer2] --read6 BUILT (fable_engineer2_v1/read6/build_read6.py; stack it after every fragment that redefines _mm): six-level READS, level 6 inside read5's packets (one level-4 slot per packet, all level-5/6 arrays inside read5's own child regions: the 3.9e8-float arena is respected), for reads with m, k, n divisible by 64 chosen by an exact steady-state bill rule dB = -0.128235 mkn + 23.447 (mk+kn) + 16.411 mn (matches in-situ measurement to the flop) + MU x platform seconds (user fit on 30 in-situ reads, residual sd 0.029 s, + 220 ops x 0.41 ms); frames of <= 12 shapes under the LRU caps. Presets on S11: read6q (MU 0.35 Gflop/s) = read6/S11_read6q.py 91641643; read6h (MU 0.15) = S11_read6h.py 1a398321. Step 1 in situ (S11, r35/r100): measured bill saving 2-3x my design price, user CPU within 18% of the model (read6/read6_shadow_s11a.json). Gates: einsum gate PASS (0 new sites), whest validate PASS (both). Parity vs S11 (OpenBLAS env, 35/100/37/39): read6q bill -0.405/-0.366/-0.370/-0.300% (pooled -0.360%), final MSE +0.04/+0.08/-0.16/+0.06% (pooled +0.001%), adjusted -0.359%, max prediction diff 2-3e-6 relative (rounding); read6h pooled bill -0.404%, MSE +0.065%, adjusted -0.339%. Paired user-CPU census read6q vs S11 queued on the porter's lock (run_usercpu_pair.sh -> platfit/optime_e2_read6q.json).
    2026-09-24 12:10 [engineer2] --read6q on S12-lite (porter's stack_S11_cp_dc35_mem.py 7333dea9 + read6 q = fable_engineer2_v1/read6/S12lite_read6q.py c20d67c7): einsum gate PASS; parity vs S12-lite (35/100/37/39): bill -0.408/-0.369/-0.373/-0.302% (pooled -0.363%, same as on S11), final MSE -0.16/+0.42/-0.17/+0.11% (pooled +0.04%, rounding-level; max prediction diff 2e-6), adjusted pooled -0.32%. Stacks cleanly after colpack + deadcut35. User-CPU pair (read6q vs S11) queued behind the porter's VALID40 lock.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/read6/build_read6.py](code/fable_engineer2_v1/read6/build_read6.py)
- [code/fable_engineer2_v1/read6/S11_read6q.py](code/fable_engineer2_v1/read6/S11_read6q.py)
- [code/fable_engineer2_v1/read6/S11_read6h.py](code/fable_engineer2_v1/read6/S11_read6h.py)
- [code/fable_porter_v1/run_usercpu_pair.sh](code/fable_porter_v1/run_usercpu_pair.sh)
- [code/fable_porter_v1/stack/stack_S11_cp_dc35_mem.py](code/fable_porter_v1/stack/stack_S11_cp_dc35_mem.py)
- [code/fable_engineer2_v1/read6/S12lite_read6q.py](code/fable_engineer2_v1/read6/S12lite_read6q.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L372](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L372)
- [TEAM_LOG_2026-09-22_to_09-26.md#L374](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L374)
- [TEAM_LOG_2026-09-22_to_09-26.md#L378](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L378)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [read6_shadow_s11a.json](read6_shadow_s11a.json) · result

</details>

## Related work

- shipped in: [Submission #332195: column packing, inactive-row selection, six-level reads, and in-place tree products](../../../submissions/332195/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-packetized-six-level-reads-eligible-shapes-save-0-363) · [Copy provenance](../../../PROVENANCE.json)

