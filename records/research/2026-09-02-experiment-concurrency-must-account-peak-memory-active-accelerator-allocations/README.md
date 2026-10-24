# Experiment concurrency must account for peak memory and active accelerator allocations

2026-09-02 · lesson · informational · measured

Concurrent full-tensor runs exhausted memory despite manageable individual runs. The revised operating rule bounded concurrent engines and worker memory using available memory, with swap treated as a trend because stale swap counts can outlast active memory pressure.

Recorded measurement:

    **REVISED 2026-09-01 (Matt):** "you can use the machine just dont OOM
    it" — full CPU/GPU utilization is authorized; the binding constraint
    is MEMORY (no OOM: keep peak RSS well under the machine's RAM and
    MPS allocations bounded; chunk large batches). The earlier
    one-job/nice-15/4-thread caps were for the lag incident, not a
    standing ceiling. Still: watch memory headroom before sizing
    batches, and prefer chunked streaming over giant allocations.
    
    **Concrete OOM rule (2026-09-01 evening incident):** two 1024³
    tensor engines (`run_tensor_recursion_v4.py`, ~6-8 GB each incl.
    MPS temps) plus two tracker jobs drove swap to 14.3 of 15.4 GB.
    Never run two 1024³ engines concurrently; check
    `sysctl vm.swapusage` before launching any engine, and treat swap
    above ~50% as "stop launching". Killing a chain's loop shell lets
    the current python finish while preventing the next leg.
    
    **Overnight autonomy (Matt, 2026-09-02): "going to sleep - work through the night, make sure you dont accidentally oom the box or anything stupid."** Rules while unattended: at most TWO numpy card workers (~5-8 GB each) plus at most ONE MPS engine (~6-8 GB) at any time; check `memory_pressure` (system-wide free %) before every launch and do not launch if free memory < 30%; NOTE the `vm.swapusage` figure is STALE on this box (macOS does not drain swap; it read ~9.5 GB all night at 75-83% free) — use it only as a trend, never as a gate; never run two 1024³ engines; kill any job whose RSS exceeds 12 GB; no destructive actions, no commits, board submissions PRE-AUTHORIZED by Matt (2026-09-02, "you are also preauthorized for any board submissions") — still only after panel verification + virgin gate + the meaningful bar or the 1.5e-8 bank line, with a ledger entry and the daily limit respected.

## Programs

- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)

[Research graph](../../../graph/index.html#node=2026-09-02-experiment-concurrency-must-account-peak-memory-active-accelerator-allocations) · [Copy provenance](../../../PROVENANCE.json)

