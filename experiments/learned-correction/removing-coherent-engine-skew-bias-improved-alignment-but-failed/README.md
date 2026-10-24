# Removing coherent engine-skew bias improved alignment but failed cross-network endpoint transfer

2026-09-04 · experiment · informational · measured

ALIGNMENT IS NECESSARY, NOT SUFFICIENT (rescaled engine k3 shift, frozen r35 table, operator ON, independent 4M): r35 beta 0.5 2.403e-8 (-11.1%), beta 0.35 2.301e-8 (-14.9%); r100 beta 0.5 3.362e-8 (+21%), beta 0.35 3.009e-8 (+8%) - despite alignment 0.84-0.91 on r100. The skew channel is worth little on r100 even at the truth (half-truth k3 -5%), so a k3 supply cannot carry the 2e-8 bar on r100-type nets; the variance channel must. Testing the rescaled engine third-cumulant repeated-index plane (K21) planes through the pair consumer (the variance route).

Code availability: The frozen skew-table module and retrofit instrument were recovered. A dedicated launch script for the exact tracker comparison was not identified.

Recorded measurement:

    - T-86 REMOVING THE ENGINE'S COHERENT UNDER-SCALE LIFTS ALIGNMENT TO THE BAR (engine ref k3u vs 32M truth; frozen per-layer table = truth-on-engine slope fitted on r35: 1.009 (L4) .. 1.030 (L8) .. 1.047 (L12) .. 1.060 (L14); applied HELD-OUT to r100): r100 skew-shift alignment with the true dk3 raw +0.84 / +0.84 / +0.82 / +0.81 / +0.69 / +0.56 at L4/6/8/10/12/14 -> with the frozen table +0.85 / +0.88 / +0.90 / +0.91 / +0.88 / +0.84 (own-slope +0.86-0.91); r35 in-sample +0.80 / +0.80 / +0.76 -> +0.90 / +0.91 / +0.92 at L8/12/14. So engine-class third-order content plus a universal per-layer scale reaches ~0.9 on both nets. Matt: "[quote removed]" -> three corpus reviewers launched (closure lanes, operator/observer lineage, SOL lanes). Now grading the rescaled engine shift in-card at beta 0.5 / 0.35 and the engine K21 planes (with the table) through the pair consumer for the variance channel.
    
    - T-86b ALIGNMENT IS NECESSARY, NOT SUFFICIENT (rescaled engine k3 shift, frozen r35 table, operator ON, independent 4M): r35 beta 0.5 2.403e-8 (-11.1%), beta 0.35 2.301e-8 (-14.9%); r100 beta 0.5 3.362e-8 (+21%), beta 0.35 3.009e-8 (+8%) - despite alignment 0.84-0.91 on r100. The skew channel is worth little on r100 even at the truth (half-truth k3 -5%), so a k3 supply cannot carry the 2e-8 bar on r100-type nets; the variance channel must. Testing the rescaled engine K21 planes through the pair consumer (the variance route).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/k3_table.py](code/public_g20_emission_retrofit_v1/k3_table.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L729](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L729)
- [LANE_g3_tracker.md#L730](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L730)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-removing-coherent-engine-skew-bias-improved-alignment-but-failed) · [Copy provenance](../../../PROVENANCE.json)

