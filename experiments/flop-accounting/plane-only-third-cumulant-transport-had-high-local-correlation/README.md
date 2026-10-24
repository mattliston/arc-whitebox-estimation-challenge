# Plane-only third-cumulant transport had high local correlation but failed full recursion

2026-09-04 · experiment · rejected · measured

HYBRID round 1 (r35, CP3_CAP=1024 + plane-residual ansatz): DIVERGED (mse 8e14). The residual I defined (post plane minus the bank's own post read of the OLD factors) grows to 100% by L7 because the bank is REBORN from the post plane each layer (births): the residual is the births' plane, which the bank then carries too -> double counting -> positive feedback. Also measured: util at CP3_CAP=1024 is 0.437 (full 0.600): the bank-dependent cost is ~0.16, the rest of the card's third-order cost is fixed (reads/planes).

Recorded measurement:

    - T-125 (plane_ansatz_test.py, MPS MC 1M samples, net 35, zero billed): the K21 PLANE (two-equal-index third cumulants P_ab = k3(h_a,h_a,h_b), 1024^2/layer) transported by the CORRECT ansatz  k3(z_i) = sum_a W_ai^3 T_aaa + 3 sum_{a!=b} W_ai^2 W_bi P_ab  (= 3 diag((W∘2)^T P W) + (W∘3)^T T; three GEMMs per layer) reproduces the next layer's per-unit pre-activation SKEW at corr 0.948/0.966/0.961/0.964 and scale 0.91-0.95 (L3->4, 7->8, 11->12, 14->15); residual rel 0.26-0.32 (label noise 0.03-0.12). Diagonal-only: corr 0.38-0.57 (useless, = T-120). CORRECTION TO T-119: the "plane carry deviates 400-600%" emulation (T-119b) transported the plane by CONGRUENCE (pre = W^T K21 W), which is the wrong physics (drops the W∘2 term); the plane line was closed on a wrong formula. The all-distinct part of the tensor is ~7% of the skew variance. This reopens the cheap third-order carry: plane transport 3 GEMMs/layer (~0.3% util each) instead of the 45% CP bank.
    
    - T-125b: exact index-pattern transport of the (non-symmetric) plane: corr 0.960/0.956 scale 0.87 (L7->8, L14->15) vs the symmetric-P formula 0.965/0.964 scale 0.92; both leave a rel residual 0.26-0.30 = the all-distinct part, which UNDER-predicts the skew coherently by 8-13% (a learned scale could absorb part). Expected plane-card raw before any closure ~ (0.27)^2 x 2.9e-6 ~ 2e-7 (third-order term only). Instrument hook RETRO_PLANE_ANSATZ=1 (with RETRO_PLANE_CARRY=1024) implements the transport; run on r35 with the full card otherwise.
    
    - T-125c RESULT (r35, full card with the third-order bank REPLACED by the plane-ansatz carry, transport unbilled): raw 8.43e-7 F2J-off at util 0.6005 (the bank still billed in the emulation). Carried pre-plane vs the bank-built pre-plane: rel deviation 0.46-0.50 at EVERY layer (vs 4-6x for the congruence). So the plane ansatz alone is 3.4x better than the Gaussian card (2.9e-6) and 30x worse than the bank (2.9e-8): the all-distinct third-order content (27% of the skew rms, 50% of the plane) is worth 30x in raw. A pure plane card at the util floor would score ~8e-8 adjusted (worse than the bank). Next: the HYBRID - a rank-capped bank (CP3_CAP=1024, real util) + the plane-ansatz transport of the PLANE RESIDUAL the truncated bank loses (post plane minus the bank's own post read).
    
    - T-125d HYBRID round 1 (r35, CP3_CAP=1024 + plane-residual ansatz): DIVERGED (mse 8e14). The residual I defined (post plane minus the bank's own post read of the OLD factors) grows to 100% by L7 because the bank is REBORN from the post plane each layer (births): the residual is the births' plane, which the bank then carries too -> double counting -> positive feedback. Also measured: util at CP3_CAP=1024 is 0.437 (full 0.600): the bank-dependent cost is ~0.16, the rest of the card's third-order cost is fixed (reads/planes). A hybrid needs the residual AFTER births (= the truncation loss only) - requires the bank lifecycle (births/transport/read). Cost-correct clean design: Gaussian recursion 0.03 + plane ansatz ~0.05 + a rank-256 bank for the all-distinct remainder ~0.01 = util ~0.1 -> needs raw < 1e-7; the plane alone gives 8.4e-7, the full bank 2.9e-8; whether a rank-256 bank of the all-distinct REMAINDER is enough is the open question (never measured; the cap-2048 result capped the WHOLE tensor).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/plane_ansatz_test.py](code/plane_ansatz_test.py)
- [code/diff_card.py](code/diff_card.py)

## Evidence

- [LANE_g3_tracker.md#L923](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L923)
- [LANE_g3_tracker.md#L924](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L924)
- [LANE_g3_tracker.md#L925](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L925)
- [LANE_g3_tracker.md#L926](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L926)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-plane-only-third-cumulant-transport-had-high-local-correlation) · [Copy provenance](../../../PROVENANCE.json)

