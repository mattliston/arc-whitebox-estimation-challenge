# Spectrally split state assimilation produced no gain

2026-09-01 · experiment · informational · measured

(2026-09-01): spectral-split assimilation NULL (monotone)

k=64/128/256 at lamLo 0.3/0.2/0.4: joint 0.191/0.180/0.202 ; all
below uniform lambda=0.75's 0.2056, approaching it from below as
k grows. The "drift = spectral bulk" premise is wrong for this
mismatch; uniform assimilation stands. Next fidelity lever: restore
the M2a alpha-word dropped for cost in T-2 (transport accuracy
~1-2 pts/step, compounding over 4 steps; trivial at 30x cadence).


Recorded measurement:

    ## T-8 (2026-09-01): spectral-split assimilation NULL (monotone)
    
    k=64/128/256 at lamLo 0.3/0.2/0.4: joint 0.191/0.180/0.202 — all
    below uniform lambda=0.75's 0.2056, approaching it from below as
    k grows. The "drift = spectral bulk" premise is wrong for this
    mismatch; uniform assimilation stands. Next fidelity lever: restore
    the M2a alpha-word dropped for cost in T-2 (transport accuracy
    ~1-2 pts/step, compounding over 4 steps; trivial at 30x cadence).
    

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
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t5.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tracker_t5.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L223](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L223)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-spectrally-split-state-assimilation-produced-no-gain) · [Copy provenance](../../../PROVENANCE.json)

