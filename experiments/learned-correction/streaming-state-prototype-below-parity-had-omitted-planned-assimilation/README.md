# Streaming state prototype below parity had omitted its planned assimilation step

2026-09-01 · experiment · informational · measured

v1 (2026-09-01): NULL below parity ; the assimilation step was the design, and v1 skipped it

Fast-path validated first: the 30x engines reproduce T-3 exactly
(L2 0.01642/third-cumulant Edgeworth operator (L3) 0.05781; K=128 mode-compression certificate 1.5e-3;
short t-grid + analytic tail; full trajectory fourth-cumulant Edgeworth operator (L4) 10.4% / L5 15.0%,
~4.5%/step). T-4 v1 (tracked-state difference-first supply, diag-
anchored only): corr 0.029 (L5) / 0.016 (L4) vs fitted-core parity
0.18 ; the drifted PAIR structure swamps the supply; local refits
beat exact-rooted-but-drifted state at the layers they are read. REDESIGN (T-5, the actual anchored architecture): CARD-ASSIMILATED
TRACKING ; at each layer, project the tracked state onto consistency
with the card's carried planes (deploy-available; one Gauss-Newton
step in the dictionary weights toward C_card; fits only to the card,
never truth ; deploy-legal), keep the extras (Ttail, C_extra,
alpha-structure), transport, repeat.

Recorded measurement:

    ## T-4 v1 (2026-09-01): NULL below parity — the assimilation step was the design, and v1 skipped it
    
    Fast-path validated first: the 30x engines reproduce T-3 exactly
    (L2 0.01642/L3 0.05781; K=128 mode-compression certificate 1.5e-3;
    short t-grid + analytic tail; full trajectory L4 10.4% / L5 15.0%,
    ~4.5%/step). T-4 v1 (tracked-state difference-first supply, diag-
    anchored only): corr 0.029 (L5) / 0.016 (L4) vs fitted-core parity
    0.18 — the drifted PAIR structure swamps the supply; local refits
    beat exact-rooted-but-drifted state at the layers they are read.
    REDESIGN (T-5, the actual anchored architecture): CARD-ASSIMILATED
    TRACKING — at each layer, project the tracked state onto consistency
    with the card's carried planes (deploy-available; one Gauss-Newton
    step in the dictionary weights toward C_card; fits only to the card,
    never truth — deploy-legal), keep the extras (Ttail, C_extra,
    alpha-structure), transport, repeat. The card carries the bulk at
    0.6-1.2%; the tracker carries only what the card lacks. Supply =
    [my model on the assimilated state] - [closure(card stats)].
    Cadence note: each cycle now ~30-60 min (fast engines).
    

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py](code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t2.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L137](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L137)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-streaming-state-prototype-below-parity-had-omitted-planned-assimilation) · [Copy provenance](../../../PROVENANCE.json)

