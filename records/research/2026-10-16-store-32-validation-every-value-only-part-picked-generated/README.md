# Store-32 validation: every value-only part picked on generated seeds reads about zero or worse on 32 store networks

2026-10-16T18:33:59Z · finding · informational · measured

The 32 store ranks with 2 x 32M noise-free per-unit truths were scored as independent networks (cold single predict, shared the estimator submitted as #336412 (S2tr) control) for every value-only part selected during the day. All of them read worse than on the reused S8/A80 seeds: the four-part stack STK4 at +0.021% instead of -0.828%, the pair-mean split at layers 7-9 at +0.588%, the energy-ordered join rows at +0.039%, the task-aware rows at +0.213% against them, the layer-15 row cut at +0.397% on 10 ranks, the dead cut 3.5 at +0.03% on 6. Consequence: S5-class clean builds project about 1.476e-9 (a tie with #336412) and the non-clean V-A+B about 1.470e-9, not near oliver; a part counts only when its pooled seeds-plus-store net is clearly negative.

Recorded measurement:

    STK4 (007a5a3e) n32: xnet +0.021% (SE .21; xMSE +0.371, bill -0.351), 16/32 lower, pooled with 24 seeds -0.305% (SE .16). UVP79 n30: xMSE +0.913% (SE .28, 9/30 lower), bill -0.32, xnet +0.59%, pooled +0.36%. JORD n32: +0.039% (SE .20, 16/32), pooled -0.108% (SE .14). B vs JORD n32: +0.213% (SE .113, 12/32), pooled ~+0.12%. L15RC30 n10: xMSE +0.397% (SE .16, 1/10), xnet +0.06; n11 +0.003, pooled -0.179% (SE .09). DC35 n6: xMSE +0.03 (pooled ~-0.07). S2tr error level on store = fresh seeds (1.34e-8 vs 1.33e-8).

## Programs

- [../2026-10-16-research-handoffs-completed-before-single-threaded-integration-lower-residual/lane_store_v2.sh](../2026-10-16-research-handoffs-completed-before-single-threaded-integration-lower-residual/lane_store_v2.sh)
- [grade_store_v1.py](grade_store_v1.py)
- [pool_v1.py](pool_v1.py)
- [pool_solo.py](pool_solo.py)
- [cand_decomp_jord_v1.py](cand_decomp_jord_v1.py)
- [cand_s2tr_l2h_stk4_v1.py](cand_s2tr_l2h_stk4_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4405](../../TEAM_LOG_2026-10-01_to_10-22.md#L4405)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4406](../../TEAM_LOG_2026-10-01_to_10-22.md#L4406)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4411](../../TEAM_LOG_2026-10-01_to_10-22.md#L4411)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4421](../../TEAM_LOG_2026-10-01_to_10-22.md#L4421)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4427](../../TEAM_LOG_2026-10-01_to_10-22.md#L4427)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4434](../../TEAM_LOG_2026-10-01_to_10-22.md#L4434)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4452](../../TEAM_LOG_2026-10-01_to_10-22.md#L4452)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4453](../../TEAM_LOG_2026-10-01_to_10-22.md#L4453)

<details>
<summary>All result files and records (2)</summary>

- [CONF24_stk4_v1.json](CONF24_stk4_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← measured by: [Four value-only changes on 49 public networks: MSE +0.318%, bill -0.541%, adjusted -0.223%](../2026-10-16-four-value-only-changes-49-public-networks-mse-0/README.md)
- ← rejects: [Task-aware Nystrom join rows (kink-weighted energy): -0.455% on 8 seeds, +0.213% on 32 store networks, removed](../../../experiments/join-sketch/task-aware-nystrom-join-rows-kink-weighted-energy-0/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-store-32-validation-every-value-only-part-picked-generated) · [Copy provenance](../../../PROVENANCE.json)

