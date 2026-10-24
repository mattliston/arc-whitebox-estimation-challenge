# Float16 provides no metered-FLOP reduction: its rate equals float32

2026-10-10T16:23Z · lesson · rejected · derived

Flopscope's dtype rate table bills float16 at 1.0 = float32 (float64 at 2.0), so no precision trick lowers the metered FLOPs; the candidate generation 40 (S40) census puts the young plane read, bank push and old tier at 78% of the metered FLOPs.

Recorded measurement:

    float16 1.0 = float32, float64 2.0; S40 census on 651: read 27.6%, push 27.1%, old tier ~23%, sandwich 9.2%, newborn 4.2%, non-product 7.7%

## Programs

- [../../../experiments/flop-accounting/non-product-flop-census-exact-request-neutral-changes-offer/code/stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py](../../../experiments/flop-accounting/non-product-flop-census-exact-request-neutral-changes-offer/code/stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L666](../../TEAM_LOG_2026-10-01_to_10-22.md#L666)

<details>
<summary>All result files and records (2)</summary>

- [s40_651.json](../../../submissions/333559/s40_651.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [FLOP census: exact kernel changes each save at most 0.2%; live-floor push cut adds 13% error](../../../experiments/flop-accounting/flop-census-exact-kernel-changes-each-save-most-0/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-float16-provides-no-metered-flop-reduction-rate-equals-float32) · [Copy provenance](../../../PROVENANCE.json)

