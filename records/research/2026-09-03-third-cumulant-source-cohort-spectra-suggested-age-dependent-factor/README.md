# Third-cumulant source-cohort spectra suggested age-dependent factor ranks

2026-09-03 · finding · informational · measured

AGE-SCHEDULED COHORT TRUNCATION (bank dumps r35; P/Q ranks by cohort age): schedule A (P 512/256/256/192.., Q 1024/768/512/384/256..): third-cumulant repeated-index plane (K21) read err 7.6% (fourth-cumulant Edgeworth operator (L4)) / 2.7% (L8) / 1.8% (L12) / 1.6% (L14), k3 5.8/1.9/1.3/1.1%, cost 10/16, 15/32, 19/48, 22/56 matrix-matrix multiplication (GEMM)-eq; schedule B (P 512/384/256/256/192.., Q 1024/1024/768/512/384/256): 4.6 / 1.8 / 1.0 / 0.9% (k3 3.5/1.2/0.7/0.7%), cost 12.5/16, 17/32, 22/48, 24/56; schedule C (tighter): 2.7 / 1.1 / 0.5 / 0.3% at 15/16, 21/32, 27/48, 30/56. The deep layers dominate the cost, so the carrier ~0.27 -> ~0.12-0.14 (B) with read errors 1-2% at depth (the card's own K21 error is 8-13%). Card estimate: 0.60 -> ~0.45 (with f32 covariance) at the deployed raw if the in-card recursion tolerates the truncation -> adjusted ~1.3e-8 (-26% vs the bank).

Recorded measurement:

    - T-95 THE COMPRESSION ESCAPE HAS A SPECTRUM (r35, card gates, cohort factor X_b(l) = M_{b->l} (g_b * C_b), map M_{b->l} = prod of W^T G): rank for 90/99/99.9% of the energy - X: age 1 138/302/461, age 2 115/250/381, age 4 86/186/284, age 6 70/150/228, age 8 58/125/191, age 10 50/108/166 (cohort L2; L4/L6 cohorts similar or lower); M: age 2 301/525/676, age 4 161/313/437, age 6 111/223/322, age 8 83/172/253, age 10 68/142/212. Products of gated random matrices decay steeply, so old cohorts of the CP3 bank live in low-rank subspaces. Per-cohort low-rank form (P_b = U_p R_p, Q_b = U_q R_q; push only the U's; read ((U_p R_p)o(U_p R_p)) Lambda R_q^T U_q^T): ~(2 r_x + 3 r_y)/1024 GEMM-eq per cohort-layer vs 4 -> the carrier 0.27 -> ~0.08-0.10 at 99% energy. Nothing is dropped (T-51's partial-removal catastrophe was dropping coherent terms), only truncated in singular directions. If the K21 planes hold at <= 5% error: card ~0.40 util at raw 2.94e-8 -> adjusted ~1.2e-8 (-34% vs the bank), and with CP4 compressed the same way ~0.33 -> ~9.7e-9. Decisive test: offline rank-truncation of the dumped bank cohorts vs the exact read.
    
    - T-95b UNIFORM-RANK TRUNCATION OF THE BANK COHORTS (bankcp_r35 dumps L4/8/12/14; cohort_lowrank_study.py; exact read reproduced to 7e-6): the P factors keep 99.9%+ of their energy at rank 256 from age 2, but the Q factors of the YOUNGEST cohorts do not (age-1 Q = W^T G, full rank: 62% at rank 256), and the K21 read error at uniform rank is set by them: rank 256 -> 15% (L14) / 36% (L4); rank 512 -> 6.4% / 15%. Age-dependent schedule needed (young cohorts near full rank, old ones 192-256); the old cohorts dominate the bank's cost, so the saving survives.
    
    - T-95c AGE-SCHEDULED COHORT TRUNCATION (bank dumps r35; P/Q ranks by cohort age): schedule A (P 512/256/256/192.., Q 1024/768/512/384/256..): K21 read err 7.6% (L4) / 2.7% (L8) / 1.8% (L12) / 1.6% (L14), k3 5.8/1.9/1.3/1.1%, cost 10/16, 15/32, 19/48, 22/56 GEMM-eq; schedule B (P 512/384/256/256/192.., Q 1024/1024/768/512/384/256): 4.6 / 1.8 / 1.0 / 0.9% (k3 3.5/1.2/0.7/0.7%), cost 12.5/16, 17/32, 22/48, 24/56; schedule C (tighter): 2.7 / 1.1 / 0.5 / 0.3% at 15/16, 21/32, 27/48, 30/56. The deep layers dominate the cost, so the carrier ~0.27 -> ~0.12-0.14 (B) with read errors 1-2% at depth (the card's own K21 error is 8-13%). Card estimate: 0.60 -> ~0.45 (with f32 covariance) at the deployed raw if the in-card recursion tolerates the truncation -> adjusted ~1.3e-8 (-26% vs the bank). Not the 4e-9 class by itself (stars 0.14 + covariance 0.03 + chain/CP4 >= 0.1 remain); the first real cost lever since the 0.6065 card. Next: implement the low-rank cohort bank in the retro instrument (RETRO_LOWRANK_CP3) and measure raw F2J on/off on r35/r100.

## Programs

- [cohort_lowrank_study.py](cohort_lowrank_study.py)

## Evidence

- [LANE_g3_tracker.md#L768](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L768)
- [LANE_g3_tracker.md#L769](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L769)
- [LANE_g3_tracker.md#L770](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L770)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-03-third-cumulant-source-cohort-spectra-suggested-age-dependent-factor) · [Copy provenance](../../../PROVENANCE.json)

