# Dictionary transport worked but omitted nonlinear readout spillover

2026-09-01 · experiment · informational · measured

(2026-09-01): dictionary transport WORKS; spillover is the gap

run_tracker_t1.py (rank35): L1 carried C 2.2% (exact frame itself:
0.75% ; the order<=2 word-Gram truncation costs 1.4 pts); after ONE
dictionary transport, L2 carried C **7.4%** (card: 0.6%). The
mixing algebra is exactly closed and matrix-matrix multiplication (GEMM)-class (2 dicts at L2). The one-step loss is the DROPPED SPILLOVER: B's sym(a,Ma)-class
words and the T-sector.

Recorded measurement:

    ## T-1 (2026-09-01): dictionary transport WORKS; spillover is the gap
    
    run_tracker_t1.py (rank35): L1 carried C 2.2% (exact frame itself:
    0.75% — the order<=2 word-Gram truncation costs 1.4 pts); after ONE
    dictionary transport, L2 carried C **7.4%** (card: 0.6%). The
    mixing algebra is exactly closed and GEMM-class (2 dicts at L2).
    The one-step loss is the DROPPED SPILLOVER: B's sym(a,Ma)-class
    words and the T-sector. EXTENSION (next cell): dictionary-PAIR
    weights — M_c = sum_{d,d'} D_d^T diag(w_c^{dd'}) D_{d'}; the
    (Ma)-vectors join as their own dictionary; cross-terms are diagonal
    in the unit index; Grams between dictionary pairs keep everything
    GEMM-class; growth 2-3 dicts/layer (fine to L16). Then re-measure
    the one-step fidelity; iterate toward gate T1 (L5, supply corr
    >=0.6 vs banked R). Package track in parallel: L6 gate landed at
    the parity band (0.1775) — fifth layer at the same level; its solo
    consumption value queues with the L3/L7 fills.
    

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t1.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t1.py)
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

- [LANE_g3_tracker.md#L84](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L84)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-dictionary-transport-worked-but-omitted-nonlinear-readout-spillover) · [Copy provenance](../../../PROVENANCE.json)

