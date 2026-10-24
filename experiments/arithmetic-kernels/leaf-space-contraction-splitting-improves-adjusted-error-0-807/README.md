# Leaf-space contraction splitting improves adjusted error 0.807% on all 40 validation networks

2026-09-25T15:32Z · experiment · adopted · measured

Native attribution isolated the splitting package from the birth-leg and reuse changes. Raw error was neutral, while the work reduction transferred across every validation network.

Recorded measurement:

      E-74 (engineer): v4 passes all gates (residual cold +0.005 s; bill -0.93% mean; MSE ratios 0.997-1.002; user CPU +5.15%; expected worst 73.7 s; setup 2.14-2.18 s quiet, 3.507 s under 10 hogs with 419 bound but only 90 fused before the guard).
    2026-09-25 09:09 porter: S26 PASSES the VALID40 bar vs S24: adjusted (pooled): -0.942%  95% CI [-1.014%, -0.868%]  wins 40/40; per-net min -1.49% median -0.91% max -0.21% | raw raw (pooled MSE ratio): +0.058%  95% CI [-0.010%, +0.128%]  wins 16/40 (ties 0); residual gate next.
    - 2026-09-25 09:10 lead: S26 VALID40 PASS (porter; vs the S24 rows): adjusted -0.942% [CI -1.014, -0.868], 40/40; raw +0.058% [CI -0.010, +0.128] (neutral); util -1.000%. Attribution vs S24+j3+treuse: the k-split part adj -0.807%, raw +0.015%, util -0.821%. S26's residual gate + staging next; S27 (S26 + arm A) VALID40 next on the lock.
    2026-09-25 09:21 porter: S26 VALID40 vs S24 rows PASS: adjusted -0.942% [-1.014, -0.868] 40/40; raw +0.058% [-0.010, +0.128] 16/40; util -0.99%. Attribution vs the S25 j3+treuse rows: k-split v4 + re-derived key list = adjusted -0.807% [-0.902, -0.715] 40/40, raw +0.015%. S27 VALID40 started 09:09:37 (lock); S26 residual gate queued behind it. Engineer told (their heavy cap gates wait on S27 VALID40 start).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/e74_ksprod_bench.py](code/fable_engineer_v1/e74_ksprod_bench.py)
- [code/fable_engineer_v1/e74_parity_v2.py](code/fable_engineer_v1/e74_parity_v2.py)
- [code/fable_engineer_v1/ksplit_fragment_v4.py](code/fable_engineer_v1/ksplit_fragment_v4.py)
- [code/fable_porter_v1/stack/stack_S26w_mem.py](code/fable_porter_v1/stack/stack_S26w_mem.py)
- [code/fable_porter_v1/valid40_vs_ref_ci.py](code/fable_porter_v1/valid40_vs_ref_ci.py)
- [code/fable_refit_v1/paired_eval.py](code/fable_refit_v1/paired_eval.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1012](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1012)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1038](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1038)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1039](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1039)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1047](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1047)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332295: third birth-term row selection, split products, frame reuse, and learned refit](../../../submissions/332295/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-leaf-space-contraction-splitting-improves-adjusted-error-0-807) · [Copy provenance](../../../PROVENANCE.json)

