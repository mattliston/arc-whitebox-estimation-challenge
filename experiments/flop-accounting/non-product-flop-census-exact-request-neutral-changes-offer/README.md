# Non-product FLOP census: exact request-neutral changes offer at most small savings

2026-10-10T03:30Z · experiment · rejected · measured

The largest non-product items are the head's dense matmuls (1.0%), the old tier's QR at the joins (0.46%), the pair map's series (0.8%) and the loop's elementwise passes (0.7%); the metered FLOPs is at its floor for the present moment-closure estimator and metered FLOPs and requests trade at about 10,000 requests per 1% of metered FLOPs.

Recorded measurement:

    head matmuls 1.0%, join QR 0.46% (Cholesky QR would save 0.03%), pair-map series 0.8%, elementwise 0.7%; Cholesky sandwich 1.84 Gflop a layer vs 1.72 today

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/census/census2.py](code/census/census2.py)
- [code/stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py](code/stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L508](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L508)
- [TEAM_LOG_2026-10-01_to_10-22.md#L538](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L538)

<details>
<summary>All result files and records (3)</summary>

- [s40_651.json](../../../submissions/333559/s40_651.json) · result
- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · explainer
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-non-product-flop-census-exact-request-neutral-changes-offer) · [Copy provenance](../../../PROVENANCE.json)

