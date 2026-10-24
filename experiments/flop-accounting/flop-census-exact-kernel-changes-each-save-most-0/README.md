# FLOP census: exact kernel changes each save at most 0.2%; live-floor push cut adds 13% error

2026-10-10T16:50Z · experiment · rejected · measured

Attributing 263 Gflop on network 651: young read 27.5%, young push 25.3%, old tier 19.7%; kernel prices -0.16%, transform reuse -0.05% and the push's dead-side row cut at gate 1e-3 -0.09% were the only exact items; a magnitude cut of the young read's columns is closed because the columns are evenly loaded.

Recorded measurement:

    99.9% of column energy needs 88-97% of columns; Karstadt-Schwartz alt basis already our kernel (12 additions per level); push row cut at the live floor +13% error for -0.88% bill; gate 1e-3 cut -0.091% bill, error +0.02%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/a51_readcut/probe_readcut.py](code/a51_readcut/probe_readcut.py)
- [code/a51_readcut/probe_trreuse.py](code/a51_readcut/probe_trreuse.py)
- [code/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmH_r2_HC@jmHh_BT@k0f8e4_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmH.py](code/stack/candidate_T_C_B_F_D_Hfull_HV2_F4Cv2t0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv6c0.8597296897041633,0.8597296897041633_EcutSv2_L3_L3o_SS0_NPv9@jmH_r2_HC@jmHh_BT@k0f8e4_Phfull_v1_CS_Rlvfuse_sized32_Rtsr_off__jmH.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L668](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L668)
- [TEAM_LOG_2026-10-01_to_10-22.md#L669](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L669)
- [TEAM_LOG_2026-10-01_to_10-22.md#L671](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L671)
- [TEAM_LOG_2026-10-01_to_10-22.md#L672](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L672)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1094](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1094)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1095](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1095)

<details>
<summary>All result files and records (2)</summary>

- [s45_651.json](s45_651.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Float16 provides no metered-FLOP reduction: its rate equals float32](../../../records/research/2026-10-10-float16-provides-no-metered-flop-reduction-rate-equals-float32/README.md)
- enables: [Request caches and seven-level strip pushes adopted: byte-identical request savings and about -0.18% FLOPs](../../../records/research/2026-10-11-request-caches-seven-level-strip-pushes-adopted-byte-identical/README.md)
- enables: [FLOP census: two Winograd leaf products consume 67%; float64-rate work is only 0.56%](../flop-census-two-winograd-leaf-products-consume-67-float64/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-flop-census-exact-kernel-changes-each-save-most-0) · [Copy provenance](../../../PROVENANCE.json)

