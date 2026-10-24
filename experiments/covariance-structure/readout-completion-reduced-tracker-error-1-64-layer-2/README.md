# Readout completion reduced tracker error to 1.64% at layer 2 and 5.79% at layer 3

2026-09-01 · experiment · informational · measured

(2026-09-01): readout completion lands ; L2 1.64%, third-cumulant Edgeworth operator (L3) 5.79%

run_tracker_t3.py (T-2 + the certified >=3 completion: exact per-
unit Var via CF-only scalars, copula-shaped tail on the order-2 post
correlations, carried as a pushed C-extra buffer):
  L2 7.40% -> **1.64%** | L3 12.8% -> **5.79%** (~4.1%/step
  accumulation remains; sources: alpha's dropped M2a word, B's
  higher words, the completion's approximate shape, no law-feedback
  of C_extra). Carried-C is now within ~2.2x of the card at L2 from
  the WEIGHTS ALONE. T-4 DESIGN (the anchored synthesis): the deployable form re-anchors
on the card's carried planes EVERY layer (no compounding, per the
anchored-delta law) while the tracker carries forward only the
EXTRAS the card provably lacks (the >=3 cross-information, T-sector,
exact-variance splits) from the exact shallow layers.

Recorded measurement:

    ## T-3 (2026-09-01): readout completion lands — L2 1.64%, L3 5.79%
    
    run_tracker_t3.py (T-2 + the certified >=3 completion: exact per-
    unit Var via CF-only scalars, copula-shaped tail on the order-2 post
    correlations, carried as a pushed C-extra buffer):
      L2 7.40% -> **1.64%** | L3 12.8% -> **5.79%** (~4.1%/step
      accumulation remains; sources: alpha's dropped M2a word, B's
      higher words, the completion's approximate shape, no law-feedback
      of C_extra). Carried-C is now within ~2.2x of the card at L2 from
      the WEIGHTS ALONE.
    T-4 DESIGN (the anchored synthesis): the deployable form re-anchors
    on the card's carried planes EVERY layer (no compounding, per the
    anchored-delta law) while the tracker carries forward only the
    EXTRAS the card provably lacks (the >=3 cross-information, T-sector,
    exact-variance splits) from the exact shallow layers. The supply =
    [one-step push of card-state + tracked extras] - [card's own next]
    — the G3-engine-delta at parity 0.18 is this WITHOUT extras; the
    tracked extras are the lever above parity. T1-gate trajectory: the
    unanchored carry reaches L5 at ~13% (6x short); the anchored form
    needs only one-step fidelity in the card's error direction.
    

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t3.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t3.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py](code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [LANE_g3_tracker.md#L116](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L116)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-readout-completion-reduced-tracker-error-1-64-layer-2) · [Copy provenance](../../../PROVENANCE.json)

