# Hub quartic ages 3-4 extended to layers 8-15: -0.513% mean-log MSE on 24 networks

2026-10-13T10:37Z · decision · adopted · measured

Extending the hub quartic of ages 3-4 from the final layer into the inner late layers passed at 24 with the out-of-selection 16 agreeing; the two bands are additive; strength 0.75 / 1.0, an L4-7 band and a stronger final layer all sit at the optimum; the native route costs +0.052% of metered FLOPs.

Recorded measurement:

    C (L8-15) mean log -0.513% (se 0.117, 19/24), OOS 16 -0.535%; A (L12-15) -0.218%, B (L8-11 + 15) -0.293%; T142 scan 0.75 -0.052%, 1.0 +0.012%, L4-7 +0.013%, final 0.75 +0.020%; native D 0.938 / 0.937, bill +0.052%, requests +182

## Programs

- [t141_table.py](t141_table.py)
- [t142_table.py](t142_table.py)
- [fragment_k4li_v1.py](fragment_k4li_v1.py)
- [k4l_grad_t_v1.py](k4l_grad_t_v1.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_hermite_group_views_QR_HELD_v2.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_hermite_group_views_QR_HELD_v2.py)
- [../../../submissions/336935/estimator.py](../../../submissions/336935/estimator.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_rtt2_request_repair_HELD_v2.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_rtt2_request_repair_HELD_v2.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_prefix22_uvp_split_HELD_v1.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_prefix22_uvp_split_HELD_v1.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_yr7_HELD_v1.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_yr7_HELD_v1.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/build_replay_patch_v2.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/build_replay_patch_v2.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [../../../submissions/336867/estimator.py](../../../submissions/336867/estimator.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_S2tr_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2022](../../TEAM_LOG_2026-10-01_to_10-22.md#L2022)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2039](../../TEAM_LOG_2026-10-01_to_10-22.md#L2039)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2040](../../TEAM_LOG_2026-10-01_to_10-22.md#L2040)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2049](../../TEAM_LOG_2026-10-01_to_10-22.md#L2049)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2050](../../TEAM_LOG_2026-10-01_to_10-22.md#L2050)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2051](../../TEAM_LOG_2026-10-01_to_10-22.md#L2051)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2056](../../TEAM_LOG_2026-10-01_to_10-22.md#L2056)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2057](../../TEAM_LOG_2026-10-01_to_10-22.md#L2057)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2065](../../TEAM_LOG_2026-10-01_to_10-22.md#L2065)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2070](../../TEAM_LOG_2026-10-01_to_10-22.md#L2070)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2120](../../TEAM_LOG_2026-10-01_to_10-22.md#L2120)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2121](../../TEAM_LOG_2026-10-01_to_10-22.md#L2121)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2149](../../TEAM_LOG_2026-10-01_to_10-22.md#L2149)

<details>
<summary>All result files and records (4)</summary>

- [ABSORB.json](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/ABSORB.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [t141_table.out](t141_table.out) · result
- [K4LI_DECISION](K4LI_DECISION) · record

</details>

## Related work

- ← enables: [Final-layer hub quartic ages 3-4 at strength 0.5 adopted: -0.247% mean-log MSE on 24 networks](../2026-10-13-final-layer-hub-quartic-ages-3-4-strength-0/README.md)
- enables: [Additional fourth-order cohort ages and overlapping strengths rejected after the layers 8-15 quartic extension](../../../experiments/fourth-order-terms/additional-fourth-order-cohort-ages-overlapping-strengths-rejected-after/README.md)
- enables: [Refit with absorbed late-layer quartic terms accepted; combined candidate seals at -0.2155%](../2026-10-13-refit-absorbed-late-layer-quartic-terms-accepted-combined-candidate/README.md)
- enables: [Third-order row-factor gradient fails out of sample because gradient replay freezes carrier states](../../../experiments/evaluation-methodology/third-order-row-factor-gradient-fails-out-sample-because/README.md)
- shipped in: [Submission #335782: four-arm refit and hub quartic ages 3-4 extended through layers 8-14](../../../submissions/335782/README.md)
- ← enables: [Native routes add pair-map skew, kurtosis, kink-plane and per-layer correction dials](../../../instruments/native-routes-add-pair-map-skew-kurtosis-kink-plane/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-hub-quartic-ages-3-4-extended-layers-8-15) · [Copy provenance](../../../PROVENANCE.json)

