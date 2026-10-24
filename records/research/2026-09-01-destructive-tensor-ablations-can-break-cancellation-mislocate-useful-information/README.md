# Destructive tensor ablations can break cancellation and mislocate useful information

2026-09-01 · lesson · informational · measured

Removing distinct third-order content broke subsequent plane transport and produced a misleading failure cascade. Endpoint consumption tests showed that a lower bond-order construction retained most of the useful correction, so the failed ablation did not prove a need for the full tensor.

Recorded measurement:

    Measured 2026-09-01 (receipts: LANE_g3_tracker.md T-12…T-18, FASTER_PATH_MEMO.md, optbeta_sweep_all_variants.log, CAP_SCREEN_*.json). CORRECTED the same night after Matt's pushback ("you are wrong that there is no path").
    
    - Consumption betas were stale ~2×; on the deploy-frame supply β_v≈0.40. Deployed card #328622 carries only the learned F2J operator → the engine supply is additive: v-only −10.6 (r35) / −15.3 (r100) / −7.8 (r150) raw. k3/k4 arms flip sign between nets → v-only.
    - WRONG first reading (retracted): "value needs the full 1024³ carried tensor, 42× budget, flat spectrum, no path." The tensor-killed / aux / Tucker-256 collapses are a CASCADE artifact of the wedge (zeroing distinct content breaks the plane push; next pre-plane 4.5× off-scale), not a measurement of where the value is.
    - RIGHT reading: `_d2` (bond order 2) = +9.8 vs full +10.3 → the born distinct content that matters is the order-2 path word Σ_m λ_m (a∘ρ_{:m})⊗e_m⊗(a∘ρ_{:m}) — exactly the card's CP3 birth (p = gate∘C[:,c], q = e_c, cap 16384 = full birth, non-binding). The card ALREADY carries that content exactly. Remaining differences engine-vs-card: (1) the engine's validated PLANE EMISSION (`emission_K21`: M21/M11 Mehler stack + Edgeworth, inputs m,v,k3,k4,C,K21,K22,K31 — all in-card), (2) exact dense plane push, (3) marginal-only reads in the card. All 1024²-class.
    - Carrier criterion still true: any per-layer plane must be ≲ the closure's own ~7% error (supply = engine plane − host plane; truth − host ≈ 7% of plane).
    - Util side: cap ladder explodes raw (8192 → +351%); plain MC at budget 1.03e-6.
    
    **Why:** three days graded supplies by pushed-v correlation; the optimal-β consumption value is the compass, and the wedge-cascade misread nearly killed a live lead.
    
    **How to apply:** (1) grade every supply with `consumption_betas.py` at optimal β first; (2) never infer "value location" from a branch that breaks the plane/distinct cancellation; (3) the climb = retrofit/parallel the engine's plane emission into the card (sub-problems in LANE_g3_tracker T-19+), then raise plane accuracy input by input (ORACLE_FOURTH etc.) toward the v-ceiling (exact v = −54%). Related: [[closed-forms-need-numerical-certificates]], [[be-gentle-with-the-machine]], [[stop-defaulting-to-defeatist]].
    
    **The cheap compression of the whole third order (re-found 2026-09-01 late, T-21):** the C1 endpoint ADJOINT (LANE_h2_h3 §1, 08-31; receipt C1_ADJOINT_RANK_rank0035.json): S = Σ_l ⟨dK21_l, M_l⟩ with closed-form M_l; value-rank ~10-150 reads/layer. It was shelved because the reads seemed to need the tensor (C3 randomized-read FAIL). Tonight's reframing supplies the reads from the emission delta on the card's own planes → deploy form = card + E_l + adjoint reads. Never re-derive this; it is in MECHANISM_TREE.md N1.2g.

## Programs

- [../../../experiments/error-attribution/complete-tangent-injections-true-supply-reached-raw-7-09e/code/public_g19_attribution_v1/build_gate1_supplies.py](../../../experiments/error-attribution/complete-tangent-injections-true-supply-reached-raw-7-09e/code/public_g19_attribution_v1/build_gate1_supplies.py)
- [../../../experiments/error-attribution/complete-tangent-injections-true-supply-reached-raw-7-09e/code/public_g19_attribution_v1/consumption_betas.py](../../../experiments/error-attribution/complete-tangent-injections-true-supply-reached-raw-7-09e/code/public_g19_attribution_v1/consumption_betas.py)
- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)

## Evidence

- [build_gate1_supplies.py](../../../experiments/error-attribution/complete-tangent-injections-true-supply-reached-raw-7-09e/code/public_g19_attribution_v1/build_gate1_supplies.py)

[Research graph](../../../graph/index.html#node=2026-09-01-destructive-tensor-ablations-can-break-cancellation-mislocate-useful-information) · [Copy provenance](../../../PROVENANCE.json)

