# Additional spillover dictionary failed to improve the readout

2026-09-01 · experiment · informational · measured

(2026-09-01): spillover dictionary NULL ; the gap is the READOUT

run_tracker_t2.py: L2 7.40% (T-1: 7.38% ; the sym(a,Ma) pair added
nothing); third-cumulant Edgeworth operator (L3) 12.8% (~5.4%/step compounding); 5 dicts/5 pairs at L3,
algebra exact and matrix-matrix multiplication (GEMM)-class. The one-step C loss is NOT transported
words ; it is the READOUT truncation: the carried-C model stops at
order-2 Grams, missing the >=3-order pair content (the same
completion that took the stage-1 absolute build from 3.1% to 0.28%:
T-birth Grams + copula tail pinned to the exact per-unit residual). NEXT (T-3): wire the certified >=3 readout completions into the
tracker's C-model; re-measure L2/L3.

Recorded measurement:

    ## T-2 (2026-09-01): spillover dictionary NULL — the gap is the READOUT
    
    run_tracker_t2.py: L2 7.40% (T-1: 7.38% — the sym(a,Ma) pair added
    nothing); L3 12.8% (~5.4%/step compounding); 5 dicts/5 pairs at L3,
    algebra exact and GEMM-class. The one-step C loss is NOT transported
    words — it is the READOUT truncation: the carried-C model stops at
    order-2 Grams, missing the >=3-order pair content (the same
    completion that took the stage-1 absolute build from 3.1% to 0.28%:
    T-birth Grams + copula tail pinned to the exact per-unit residual).
    NEXT (T-3): wire the certified >=3 readout completions into the
    tracker's C-model; re-measure L2/L3. Note the compass: carried-C
    rel is the development diagnostic; the T1 GATE is the
    difference-first SUPPLY corr vs banked R (pushed-v), which can be
    better than the absolute carry (matched-model cancellation).
    

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t2.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py](code/p2_width1024_depth16/public_g19_attribution_v1/fast_engines.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t3.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t3.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L101](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L101)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-additional-spillover-dictionary-failed-improve-readout) · [Copy provenance](../../../PROVENANCE.json)

