# Backend-time paired profiler resolves an underestimated padded-depth compute cost

2026-09-23T15:36Z · tool · informational · measured

Matched public telemetry showed a real kernel cost increase beyond the server-CPU proxy. Summed backend operation wall time tracked that increase more closely, while heavy-load six-network gates retained substantial noise.

Recorded measurement:

    2026-09-23 08:36 [engineer2] S7 PLATFORM TELEMETRY (eval_332026.json read 08:26, 46/100 MLPs done; paired by MLP index vs S6b #332005, fable_engineer2_v1/telemetry_compare.py): wall +4.77 s mean / +5.23 s median per MLP, server kernel +6.7%, util -1.52%. The excess holds in every index block (+3.5..+6.1 s); no worker re-setups so far. S7's worst walls are now WARM MLPs at 101.5-102.9 s (S6b's warm max 99.1 s). My gate projected +1.54 s (local paired server CPU +2.72%): local CPU under-read the platform kernel on both recent fires (S6b: local -2.1% vs platform +0.4%; S7: local +2.7% vs platform +6.7%). Open question: does the platform penalize deeper Winograd and batched small products (push5p6, --batch) more than the local OpenBLAS server, or was this run's host uniformly slower? The remaining 54 MLPs and the analyst's per-run flops+cold fit can tell. S8 implication: the engineer's cold r35 census (+2.3% local CPU, +2.6k round trips, util -1.02%) projects about +2.7 s per MLP on the old calibration and about +5 s if the platform adds ~3 points again (expected worst ~104 s, over the 100 s rule). My S8 paired gate vs S7 is running (in-process ~08:40, paired 20 nets ~09:30).
    2026-09-23 08:52 [engineer2] CORRECTION to my 08:36 S7 telemetry line and my message to the lead: the "slower host" evidence was wrong. Paired by MLP index, S7's residual is flat (-0.9 ms on MLPs 0-45, +1.0 ms on 0-61); the 0.25 vs 0.26 s I quoted compared rounded, unpaired means. Paired server overhead -0.59 s (-7%) for a -9.9% round-trip cut fits an equally fast server (within ~3%). At 62/100 MLPs: wall +5.07 s/MLP, kernel +6.9% (+7.55% on MLPs 46-61), util -1.37%, max wall 103.0 s, no re-setups, no failures. S7's expected worst is ~94.7 + 5.1 = ~99.8 s. So the platform charges S7's kernel changes about 2.5x the local server CPU (S6b's --batch showed the same bias: local -2.1%, platform +0.4%). S8 (symout v3: local +2.3% CPU, +2.6k requests) then projects about +5.3 s/MLP, expected worst ~105 s: I advise not firing S8 on local bars alone.
    2026-09-23 08:36 [engineer2] Timestamp fix: my CORRECTION entry above stamped 08:52 was written at ~08:35 (hand-stamped, wrong); the 08:36 S7 telemetry entry was written at ~08:31. Content unchanged.
    2026-09-23 09:11 [engineer2] PLATFORM CALIBRATION (fable_engineer2_v1/platfit/): S7's +8% platform kernel per flop is NOT explained by (1) its billed op mix: models on deterministic local features (GEMM flops, other billed flops, calls; fit_platform_v2.py) fitted on S2..S6b under-predict S7 by 7-9% (S7 vs S6b: GEMM flops -1.7%, other +3.4%, calls flat); nor (2) leaf-matrix count (leaf_census.py, r100): S7 moves ~45 Gflop of leaf GEMM from depth 4 to depth 5 (depth-5 matrices 3.88M -> 5.76M, same 1,568 matmul calls), but S4 had MORE depth-5 matrices (4.47M) than S6b at the LOWEST platform ns/kflop (212), and S4->S5e raised ns/kflop +2.6% while depth-5 matrices fell. S8so3 has S7's depth-5 profile (+0.24M depth-4, +140 matmul calls). Discriminator: S9 (S7's exact kernel code) on the platform, ~232 ns/kflop = code, ~215 = host. Until then every kernel change is priced with both projections (old model and x2.5 amplification). S9 gate: params-only vs S7 (params_only_diff_v2.py), in-process bills identical on r35/r100/r37; paired 6-net run vs S7 in progress (load 20, ~09:30).
    2026-09-23 09:35 [engineer2] Launch-order bias check: NONE. Older near-identical paired runs read first-arm CPU -0.32% (cb3+setup vs cb3), -0.59% (S7+hold2 vs S7), +0.18% (hold2 vs bindhold), per-net spread <1%; S9's -2.47% was contention noise at load 17-20 with another agent's server pairs running. S7's local +2.7% stands; the calibration gap is real. 6-net gates under heavy load carry ~+-2.5% CPU noise: S10L's gate will run ABBA (2 x 6 nets, launch order swapped).
    2026-09-23 09:36 [engineer2] S9 #332037 EARLY TELEMETRY (13/100 MLPs; fable_engineer2_v1/telemetry/nskflop.py, paired by index): S9 = S7's exact kernel code reproduces S7's cost: mean wall +0.18 s vs S7; ns/kflop on shared indices 5-12: S9 225.8 median / 232.5 mean, S7 227, S6b 211. So S7's +8%/flop is a CODE effect (push5p6), not a slow host (preliminary). Platform noise between identical-code runs is LARGE: per-MLP wall differences sd 7.4 s (-11.1..+11.6 s); S9 MLPs 8/9 ran 110.4/110.1 s (+10.6/+11.6 s vs S7, kernel/flop +12%, a contended window on two workers). Per-MLP flops differ up to +-0.8% from MLP 7 on (different MLP-to-worker order changes cached frames). Implications: the x2.5 rule applies; S10L (symout +1.4-2.3% local CPU) projects ~103-104 s expected worst (fails); reverting push5p6 should recover ~-7% kernel (~-6 s/MLP) for ~+0.5% warm bill.
    2026-09-23 09:38 [engineer2] CORRECTION (complete S7 run, 100 MLPs paired vs S6b; telemetry/nskflop.py): kernel +5.23% mean / +4.17% median, wall +3.70 s mean / +3.20 s median, flops -1.08%. The run was uneven: MLPs 0-61 +6.7..+7.6% kernel, MLPs 62-99 +2.5% (+1.5 s). ns/kflop over 95 shared indices: S7 232.0, S6b 218.4, S5e 219.2, S4 213.7, S2 215.8 (S7 +6.2%/flop vs local +3.5%). Amplification vs my local +2.72% server CPU: ~1.9x mean, ~1.5x median (NOT 2.5x, which came from the partial 62 MLPs). S7 expected worst ~94.7 + 3.7 = ~98.4 s. S10L (symout +1.4-2.3% local CPU) at x1.9: +2.3..+3.8 s -> ~100.7-102.2 s; at x1.5: ~100.3-101.5 s (just over 100). Clean-elementwise fit (platfit/fit_platform_v3.py): platform kernel ~71% elementwise (~15x the cost per billed flop of GEMM), 28% GEMM; LOO +-5% (too coarse for 2-3% calls).
    2026-09-23 09:48 [engineer2] CALIBRATION INSTRUMENT: a CONCURRENT in-process backend-time pair (platfit/run_backend_pair.sh; per-op backend wall summed per net, the local analog of the platform kernel) gives S7 vs S6b +5.75% over 6 nets (warm +6.5% mean / +8.0% median; per-net -2.3..+17%, so ~+-3% on 6 nets), against the platform kernel +5.2% (100 MLPs) and local paired server CPU +2.7%. The consistent signal is non-GEMM time (+4..+18% on every net: push5p6's extra Winograd elementwise level). Server CPU dilutes kernel changes; proposal: price kernel changes by paired in-process backend time at x1 (enough nets), keep server CPU for request/residual effects. EXPECTED WORST (analyst's method, telemetry/expected_worst.py; reproduces S6b 94.8 s): S7 #332026 complete = 100.3 s, S5e 99.1 s, S9 #332037 at 31 MLPs 96.3 s (not final). S10' (S9 with push5p6 swapped for symout v3 + lean) gate ready: ABBA 2x6 vs S9 + backend pair vs S9, when the porter posts the file.

## Programs

- [code/telemetry_compare.py](code/telemetry_compare.py)
- [code/platfit/fit_platform_v2.py](code/platfit/fit_platform_v2.py)
- [code/platfit/leaf_census.py](code/platfit/leaf_census.py)
- [code/s9/params_only_diff_v2.py](code/s9/params_only_diff_v2.py)
- [code/telemetry/nskflop.py](code/telemetry/nskflop.py)
- [code/platfit/fit_platform_v3.py](code/platfit/fit_platform_v3.py)
- [code/platfit/run_backend_pair.sh](code/platfit/run_backend_pair.sh)
- [code/telemetry/expected_worst.py](code/telemetry/expected_worst.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L270](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L270)
- [TEAM_LOG_2026-09-22_to_09-26.md#L273](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L273)
- [TEAM_LOG_2026-09-22_to_09-26.md#L275](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L275)
- [TEAM_LOG_2026-09-22_to_09-26.md#L285](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L285)
- [TEAM_LOG_2026-09-22_to_09-26.md#L296](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L296)
- [TEAM_LOG_2026-09-22_to_09-26.md#L297](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L297)
- [TEAM_LOG_2026-09-22_to_09-26.md#L298](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L298)
- [TEAM_LOG_2026-09-22_to_09-26.md#L299](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L299)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../graph/index.html#node=2026-09-23-backend-time-paired-profiler-resolves-underestimated-padded-depth-compute) · [Copy provenance](../../PROVENANCE.json)

