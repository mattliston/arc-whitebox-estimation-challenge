# Rank-32 residual-plane transport with one power iteration improves adjusted error 5.352%

2026-09-25T01:04Z · experiment · adopted · measured

A seeded range finder with one power step and small singular-value truncation replaces two dense star products with thin products. Native accuracy, runtime and residual checks all passed.

Recorded measurement:

    - 2026-09-24 18:04 [engineer] LOW-RANK STAR, NATIVE (E-62, for S18; in process on S16r3b, nets 37/39/35/100 vs S16, 8M truth). Star plane A ~ Q B by randomized range finder (fixed seeded Gaussian Omega), push = (w^T Q) B and s2 = ((w^T o w^T) B^T)(w^T Q)^T thin, s1 = (push o w^T) w batched with the covariance's C w in ONE depth-6 pair (2 of the 4 batched 1024^3 products per layer gone). The native star plane is NOT low-rank (98% energy at rank 131-720): a plain rank-48 sketch = MSE +7.85% mean (bill ~-6%, adjusted LOSES). With power iterations the sketch matches the exact rank-32 SVD tail per layer: rank 32 + 2 iterations MSE +0.67% (37 +1.96% vs torch +2.58%, 39 +0.23% vs torch +0.53%), bill -4.6%, adj ~-4.0%; rank 48 kept + 2 iterations MSE +0.03%, bill -4.1%, adj ~-4.1%; rank 32 + 1 iteration MSE +0.90%, bill -5.5%, adj ~-4.6%. Per-net noise ~+-1%: torch VALID40 decides r. Wall: paired user-CPU pair (porter's tool) queued. PREBIND 12-net gate (S16r3bp2b vs S16r3b, same harness): max handles 28,061 vs 28,031, live identical; requests/bill: cold -3,767 / -1.67%, 2nd -1,033 / -0.39%, 3rd-5th -0.05..-0.11%, 0 after (platform average ~-0.12..-0.23% of the bill, plus the cold-MLP residual).
    2026-09-24 18:18 porter: S18 = S17p + lrstar2 k48 q1 r32 (ab8c8ba8) queued behind S17p staging (fire_chain_S18.sh; torch pre-gate j05 r32 <= +0.8%; bar adj <= -3.0%, CI excl 0, >= 30/40, raw +-1%). Census: -5.57% bill cold r100.
    - 2026-09-24 18:37 [engineer] S18 WALL GATE PASS: paired user CPU S18 (ab8c8ba8) vs S17p (9a4ccc1a), 6 nets (35,100,37,39,42,48): user -6.35% (se 0.32%; every net -5.4..-7.6%), bill -5.60% (-5.40..-5.71%), op calls -1,797/net, no page faults. project_wall_v3 on S16's page (#332210, 72.6 s): S18 expected worst 69.6 / 68.7 / 67.8 s, room 30.4-32.2 s (S17p assumed wall-neutral vs S16). Logs: fable_engineer_v1/usercpu_pair_S18.log, fable_engineer2_v1/platfit/optime_e_S18lr.json + optime_e_S17pref.json. S18 client/server identity (setup) running.
    - 2026-09-24 18:38 [engineer] S18 CLIENT/SERVER IDENTITY PASS: S18 (ab8c8ba8) with setup() on 35/100 is byte-identical (max abs diff 0.0) to my in-process S17-j05 + lrstar2_k48_q1_r32 predictions: the seeded draw, QR, small SVD and None-indexing behave identically through the numpy-free client. Setup bound 358 entries in 2.02 s. All engineer-side S18 gates PASS (validate, einsum, wall pair, identity); the porter's VALID40 bar decides.
    - 2026-09-24 19:10 lead: S18 VALID40 vs S17p PASS (engineer and porter): adjusted -5.35%, 40/40, raw +0.23%. Wall pair user CPU -6.35%, bill -5.60%, expected worst ~69.6 s. Staging after the residual gate (~19:35). Star torch VALID40 final: B2 r32 +.375%, r64 +.290%, j05 r32 +.190% (18/40). RULE: from S19 on, all refits train on the S18 card (low-rank star). The engineer writes a torch hook that matches the native factorization exactly (k=48, 1 power step, rank 32) and certifies it against native. The engineer then runs the covariance diag+low-rank test (r=128/256/512, kill if r256 > +0.5%) and profiles the refit step.
    2026-09-24 19:01 porter: S18 STAGED fable_fire_v1/fire_candS18/candS18_S17_lrq1r32_mem.py (ab8c8ba8): VALID40 vs S17p adj -5.352% [-5.598, -5.097] 40/40, raw +0.232% [-0.027, +0.498], util -5.569%; residual cold -0.025 s PASS; engineer wall pair user CPU -6.36%, identity PASS.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_porter_v1/fire_chain_S18.sh](code/fable_porter_v1/fire_chain_S18.sh)
- [code/fable_fire_v1/fire_candS18/candS18_S17_lrq1r32_mem.py](code/fable_fire_v1/fire_candS18/candS18_S17_lrq1r32_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L535](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L535)
- [TEAM_LOG_2026-09-22_to_09-26.md#L547](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L547)
- [TEAM_LOG_2026-09-22_to_09-26.md#L555](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L555)
- [TEAM_LOG_2026-09-22_to_09-26.md#L556](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L556)
- [TEAM_LOG_2026-09-22_to_09-26.md#L558](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L558)
- [TEAM_LOG_2026-09-22_to_09-26.md#L569](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L569)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [usercpu_pair_S18.log](usercpu_pair_S18.log) · log
- [optime_e_S18lr.json](optime_e_S18lr.json) · result
- [optime_e_S17pref.json](optime_e_S17pref.json) · result

</details>

## Related work

- shipped in: [Submission #332226: rank-32 memoryless (2,1) residual plane with one power iteration](../../../submissions/332226/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-rank-32-residual-plane-transport-one-power-iteration-improves) · [Copy provenance](../../../PROVENANCE.json)

