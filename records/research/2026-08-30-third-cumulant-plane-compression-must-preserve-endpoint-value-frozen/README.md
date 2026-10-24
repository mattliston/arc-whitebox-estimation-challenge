# Third-cumulant plane compression must preserve endpoint value and frozen transfer

2026-08-30 · lesson · informational · measured

Five compression families failed to preserve the variance-relevant value of the third-cumulant repeated-index plane error. A favorable single-network fit also failed frozen transfer, establishing that energy capture and in-sample correlation were insufficient acceptance tests.

Recorded measurement:

    Five compression families for the plane error dK21 (the 93% v-channel
    supply) tested against the VALUE metric (dv_corr through the tangent,
    r35, full = 0.813) on 2026-08-30 — all closed:
    - SVD rank 4/16/64/256 → 0.379/0.323/0.340/0.395 (rank 16 WORSE than
      4: energy-ranked directions anti-select value).
    - Collective rank-1 projection → −0.04 (remainder alone 0.79).
    - One-step star basis (emission spread) → 6-13% of fresh; amplitudes
      corr 0.5-0.6 with analytic emission (physics right, share tiny).
    - Host ridge shapes (β², β) → profile R² 0.04-0.12.
    - Symmetric unit permutation → 0.094: **~90% identity-specific**.
    Wedges: KILL_TENSOR (engine map, plane memory) direction 0.59 at L15
    but 5x overshoot (carried distinct content cancels fresh emission);
    FRESH_STOP=8 keeps 0.42. t=0 rank-1 Mehler chain KEEPS value (0.785)
    — consumption is simple, supply is the whole blocker.
    
    **Why:** prevents re-attempting any "summarize/learn the object"
    carrier; the content must be computed for the specific net by an
    actual third-order propagation.
    
    **08-30 evening addendum — kt REPLICATION FAIL:** the KILL_TENSOR
    plane-memory map's r35 19.3% was single-net overfit on oracle
    covariance: r100 opt-beta 6.4% vs full engine 15.1%; frozen-beta
    transfer −80% (betas swing 0.59→0.05). Plane-memory-map carriers are
    DEAD without a cross-net-stable scale/direction mechanism. Full
    engine = only stable supply (frozen ~10%, LOO 18.7%). f32 vs f16
    storage: 14.0 vs 14.1 — precision not a lever. ALWAYS run the second
    net + frozen-beta transfer before believing any single-net capture.
    
    **How to apply:** carrier work = COST REDUCTION of the full engine
    (DTOT bond-order wedge, term pruning) + deploy-billing thresholds.
    Env gates in run_tangent_v2.py: DK21_FILTER/DK21_RANK/DK21_PERM/
    INJ_LAYERS; engine: DTOT/KILL_TENSOR/FRESH_STOP/DEPLOY_FRAME (default
    frame reads TRUTH m,v,C per layer — check premise provenance).
    Related: [[chain-value-is-dv-channel]], [[tensor-recursion-instrument]].

## Programs

- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../../../experiments/error-attribution/collective-aware-tangent-added-only-0-3-0-2/code/public_g19_attribution_v1/run_tangent_v2.py](../../../experiments/error-attribution/collective-aware-tangent-added-only-0-3-0-2/code/public_g19_attribution_v1/run_tangent_v2.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)

## Evidence

- [run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)

[Research graph](../../../graph/index.html#node=2026-08-30-third-cumulant-plane-compression-must-preserve-endpoint-value-frozen) · [Copy provenance](../../../PROVENANCE.json)

