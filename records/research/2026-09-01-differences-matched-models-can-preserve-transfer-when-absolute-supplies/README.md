# Differences of matched models can preserve transfer when absolute supplies fail

2026-09-01 · lesson · informational · measured

The resolvent-based post-state supply subtracted the engine emission at matched implied statistics before it was consumed. This difference-first construction transferred positively across two networks after earlier absolute-supply approximations had failed.

Recorded measurement:

    As of 2026-09-01: the G3 supply candidate (resolvent-exact post-state
    (α,B,T) from identified (a,M)-cores, certified Wick K21post assembly
    (1,1,1,½,1), MINUS the engine's emission on matched implied stats =
    the difference-first delta; pushed diag → dv̂-augmentation with per-
    layer γ) achieves per-layer parity with the certified chain (the
    chain's own single-layer windows grade 0.15-0.18) and FROZEN-
    TRANSFERS positive: +1.1% raw in-sample (r35, L4/L5/L8), +0.6% with
    frozen γ on r100. Zero fitted parameters in the supply itself.
    
    **Why:** every prior supply candidate failed frozen transfer (twelve+
    wrong-sign results); this one holds because it is a difference of
    MATCHED models (the difference-first law) built from closed forms.
    
    **How to apply:** the continuation is all-12-layer cores (recipe:
    r=32, a-heavy init + annealed l4=0.05, NIT=20k — the a/M-split needs
    4th-order pressure; (C,K21) alone picks the unphysical M-branch),
    LOO-beta integration instead of crude γ, virgin gate, board decision
    co-fired with the f32 pack. Key files: run_g3_engine_delta.py,
    run_ncqc_corefit_joint.py, land_stack.sh/land_r100.sh in
    [[whest-repo-map]] attribution_v1; full trail in
    LANE_h2_h3_endpoint_adjoint.md. Related laws:
    [[closed-forms-need-numerical-certificates]].

## Programs

- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../../../experiments/covariance-structure/additional-spillover-dictionary-failed-improve-readout/code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py](../../../experiments/covariance-structure/additional-spillover-dictionary-failed-improve-readout/code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [../../../experiments/fourth-order-terms/chaos-state-engines-scalar-cumulant-resolvent-identities-certified-two/code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_engine_delta.py](../../../experiments/fourth-order-terms/chaos-state-engines-scalar-cumulant-resolvent-identities-certified-two/code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_engine_delta.py)
- [run_ncqc_corefit_joint.py](run_ncqc_corefit_joint.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)

[Research graph](../../../graph/index.html#node=2026-09-01-differences-matched-models-can-preserve-transfer-when-absolute-supplies) · [Copy provenance](../../../PROVENANCE.json)

