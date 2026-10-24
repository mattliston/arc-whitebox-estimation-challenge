# Billed low-rank source-cohort carrier preserved full-rank parity but compressed forms lost accuracy

2026-09-04 · experiment · rejected · measured

P-ONLY COMPRESSION DEAD (RETRO_LR_BANK="256;1024": birth factor truncated to 256 once, map never): util 0.698 (from 0.7705 plain) but raw 1.25e-7 / 1.92e-7 (the randomized range finder at birth loses the plane). COMPRESSION CLOSED WITH RECEIPTS (T-95..T-97b): the cohort spectra decay, but a billed truncation plus the dense-map reads cost more than they save; the O(depth^2) push/read of the third-order cohorts is the price of the distinct third-order content in any factor form (probe / adjoint views are transposes of the same O(L^2) maps). The instrument keeps the LR path (certified at full rank) for future free-basis ideas.

Code availability: The shared retrofit instrument retains RETRO_LR_BANK. The tracker states the tested ranks and measurements, but a dedicated run script for this exact comparison was not identified.

Recorded measurement:

    - T-97 LOW-RANK COHORT BANK IN FLOPSCOPE (RETRO_LR_BANK; _lr_push/_lr_schedule/_lr_read/_lr_k3/_lr_grow; randomized range finder = billed matmuls + fnp.linalg.qr): CERTIFICATE full rank "1024;1024" reproduces the plain-push card exactly (r35 F2J on 2.70375e-8, off 8.6218e-8, util 0.7705). SCHEDULE C BILLED: util 0.8394 (> plain 0.7705, > arena 0.60) with raw 3.79e-8 / 9.28e-8 (the range finder without power iteration is less accurate than the exact SVD emulation of T-96). Why the estimate of T-96 was wrong: (i) a cohort's read is cheaper only when BOTH factors are low-rank (P factored with Q dense costs 4 + 4 r_p/1024 > 4), and Q = the transport map is full rank for 3 ages; (ii) each truncation is billed (0.6-2.5 GEMM-eq), ~13 GEMM-eq per cohort lifetime. Retracted. Remaining variant: P-only truncation at birth (one 0.6 GEMM-eq truncation, saving 2 - 4 r_p/1024 per cohort-layer), ceiling ~25% of the carrier (~0.07 util) if raw holds; only worth it on top of the Winograd arena path (-15% adjusted at best).
    
    - T-97b P-ONLY COMPRESSION DEAD (RETRO_LR_BANK="256;1024": birth factor truncated to 256 once, map never): util 0.698 (from 0.7705 plain) but raw 1.25e-7 / 1.92e-7 (the randomized range finder at birth loses the plane). COMPRESSION CLOSED WITH RECEIPTS (T-95..T-97b): the cohort spectra decay, but a billed truncation plus the dense-map reads cost more than they save; the O(depth^2) push/read of the third-order cohorts is the price of the distinct third-order content in any factor form (probe / adjoint views are transposes of the same O(L^2) maps). The instrument keeps the LR path (certified at full rank) for future free-basis ideas.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L772](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L772)
- [LANE_g3_tracker.md#L773](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L773)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-billed-low-rank-source-cohort-carrier-preserved-full-rank) · [Copy provenance](../../../PROVENANCE.json)

