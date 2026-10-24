# Selective depth frontier identifies early shared-operand depth-seven products as a viable wall tradeoff

2026-09-25T19:12Z · experiment · informational · measured

Measured per-class alternatives are ranked by work saved per added wall time. Early shared-operand products clear the local build threshold, while later low-value classes are excluded.

Recorded measurement:

    - 2026-09-25 12:12 [engineer2] DEPTH FRONTIER PRICED (lead's selective depth-7 + depth-5 padding asks; fable_engineer2_v1/depth7/: bench_frontier.py = S27's alt-basis kernel in process, OpenBLAS env, 1 thread, every census class native vs alternative, 5 alternating reps, keys bound fused; frontier.py; bench_colpack.py = the production _mm dispatch on S28). Base backend measured in the same process: 25.8 s/net (net 100). Wall = WALL v3 on S27 #332295 (worst-MLP kernel 66.6 s: 1 local backend-s = 2.0 platform-s, plus 0.41 ms/op). Moved products carry 2.0-2.2x rounding error (rel 2.8e-5 -> 6e-5).
      Frontier per net (nets 35/100, central wall, projected from S27's 77.5 s max):
      (1) depth-5 -> 6 padded, top 16 classes: -0.22% for +0.95 s (78.4 s).
      (2) + depth-7 COLPACK, layers 2-5 (m = k = 1024; n = 2048/4096/6144/8128 as 2048-wide packets sharing ONE depth-7 A tree; exact bills on S28's dispatch: -126/-275/-425/-505 MF): -0.437% for ~+3.4 s (band 2.6-4.2): cum -0.66% at +4.4 s (81.9 s).
      (3) + the other 49 positive depth-5 classes: -0.18% for +2.8 s: cum -0.84% (84.7 s).
      (4) + depth-7 1024^3 (covariance C.w 15/net + newborn pushes 3/net): -0.305% for +3.7 s: cum -1.15% (88.4 s).
      (5) next would be 512x1024x1024 covariance (-0.066% for +2.1 s) -> 90.5 s: STOP. Everything after is <= 0.03%/s (k=384 old_reconstruct packets, 512x1024x512) or LOSES bill (384x384xn joins at depth 7).
      Depth-5 classes whose padded depth-6 trees do not fit (11 of 81 on these nets: 672x8192x864, 864x5632x384, ...) need chunking = the engineer's 11:55 lever; the depth-5 padding lever covers only whole-product moves. Coordination point: for k >= 4096 products that fit unchunked (576x4608x384, 384x6144x384, 384x5632x384), whole depth 6 beats chunking.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/depth7`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/bench_frontier.py](code/bench_frontier.py)
- [code/frontier.py](code/frontier.py)
- [code/bench_colpack.py](code/bench_colpack.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1193](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1193)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1194](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1194)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1195](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1195)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1196](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1196)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1197](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1197)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1198](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1198)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1199](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1199)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1200](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1200)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-selective-depth-frontier-identifies-early-shared-operand-depth-seven) · [Copy provenance](../../../PROVENANCE.json)

