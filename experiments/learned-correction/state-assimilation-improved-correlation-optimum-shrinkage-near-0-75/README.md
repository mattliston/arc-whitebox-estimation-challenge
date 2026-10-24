# State assimilation improved correlation, with optimum shrinkage near 0.75

2026-09-01 · experiment · informational · measured

dial complete (2026-09-01): lambda*=0.75 -> 0.166; scalar exhausted at parity's edge

Assimilation-strength curve (L5): 0->0.030 | 0.25->0.064 |
0.5->0.157 | **0.75->0.166** | 0.9->0.143 | 1.0->0.109 | hard->0.002. Single-peaked; resid-corr at peak 0.111. The scalar dial lands the
tracked supply AT the fitted-core band using only deploy-legal
inputs.

Recorded measurement:

    ## T-5 (2026-09-01): card assimilation WORKS — 0.029 -> 0.157 in one crude step
    
    run_tracker_t5.py (one A-only Gauss-Newton projection onto the
    card's carried C per layer; closure side on card stats; fully
    deploy-legal chain): L5 corr 0.1568 (T-4: 0.029; parity 0.18),
    resid-corr 0.0925 (the fitted-core band); L4 weaker (0.060/0.092).
    The transport drift was the binding error and one crude projection
    recovered ~85% of parity FROM THE TRACKED STATE — with the extras
    riding. REFINEMENT LADDER (each ~40 min): (b) iterate GN 2-3x;
    (c) assimilate pair-weights on the residual E; (d) match the card's
    K21 too; (e) diagnose L4. Gap to gate T1 (0.6): 4x — the
    assimilation response says headroom is real.
    
    
    ## T-5b (2026-09-01): OVER-assimilation erases the signal — the dial found
    
    3x GN + pair-scale: L5 corr 0.157 -> 0.0023. The supply lives in the
    residual between the tracked state and the card; hard assimilation
    collapses it. ASSIMILATION STRENGTH IS A BIAS-VARIANCE DIAL:
    lambda=0 -> drift swamps (0.03); one half-step -> 0.157; full -> 0.
    Sweeping lambda {0.25, 0.75} to bracket; the chosen lambda is ONE
    global scalar (frozen-transfer-testable like gamma — discipline-
    legal). The principled version later: split E into drift vs signal
    directions (drift = transport-error subspace, signal = the
    closure-error direction); for now the dial.
    
    
    ## T-5 dial complete (2026-09-01): lambda*=0.75 -> 0.166; scalar exhausted at parity's edge
    
    Assimilation-strength curve (L5): 0->0.030 | 0.25->0.064 |
    0.5->0.157 | **0.75->0.166** | 0.9->0.143 | 1.0->0.109 | hard->0.002.
    Single-peaked; resid-corr at peak 0.111. The scalar dial lands the
    tracked supply AT the fitted-core band using only deploy-legal
    inputs. Next rungs: (d) per-unit k3-assimilation (card k3 constrains
    cubic structure; Newton on per-unit pair-weight scale — spectra
    already in the engine pass); (e) drift/signal subspace split of E;
    (f) L4 diagnosis; (g) C_extra passthrough coefficient. T1 remains
    0.6; the ridge is climbable — every intervention has moved the
    number in the predicted direction.
    

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t5.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t5.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py](code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t2.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L156](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L156)
- [LANE_g3_tracker.md#L169](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L169)
- [LANE_g3_tracker.md#L181](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L181)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-state-assimilation-improved-correlation-optimum-shrinkage-near-0-75) · [Copy provenance](../../../PROVENANCE.json)

