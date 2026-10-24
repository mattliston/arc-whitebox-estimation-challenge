# Exact-zero leaf operands save less than 0.00009% of the complete estimator bill

2026-09-21 · finding · rejected · derived

No all-zero operand batch appeared among 17,475,569 leaf batches; even optimistically removing scalar zero terms left the cost target out of reach.

Recorded measurement:

    17,475,569 leaf batches. Removing every scalar product with an exact-zero
    operand would save at most **350,566 / 357,346 FLOPs** on networks 35 / 100
    within this fixed leaf schedule. Each is less than 0.00009% of the complete
    products for free, the optimistic total savings remain **14.759988667% /
    13.690531080%**, below the fixed **16.304320666%** uniform planning cut for
    3e-9. Both cases fail. No sparse kernel, estimator, package, or upload follows.
    | 35 | 413448696909 | 164105764518 | 175283 | 352423716100 |
    | 100 | 416264792412 | 167252818330 | 178673 | 359275931633 |
    Q=min(T,m*n) gives the lower count T+max(0,T-m*n). This is an optimistic
    All 795 charged leaf calls match the installed dense FP32 tariff and all
    four original source-site totals exactly. Every one of the 590 outer call
    match every one of the 16 original output rows and their complete cold bills.
    batching, strided/transposed inputs, permutation, a buffer larger than 1MiB,
    scalar enumeration covers 4,420 binary matrix pairs and verifies the term
    replay derives 2T-min(T,batch*m*n) and exactly matches the recorded sum of
    per-batch lower counts for every one of the 795 calls; no additional arrays,
    Both full native calls used 829,713,489,321 measured
    FLOPs. All five guarded groups retired; controller 11458 is terminal. Peak
    sampled group RSS was 6,457,212,928 bytes,
    peak process RSS 6,457,606,144 bytes, with zero added
    swap. About 0.964 MB of new
    Official 331684 remains adjusted 3.5844144212297072e-9 and raw
    1.949715441540434e-8, 100/100 cases, zero failures. The 4e-9 milestone is
    banked; 3e-9, 2e-9 and approximately 1e-9 remain unmet. The full Phase 2
    goal remains active. At unchanged raw error, the 0.1 multiplier floor rules

## Programs

- [analyze.py](analyze.py)
- [audit.py](audit.py)
- [capture.py](capture.py)
- [common.py](common.py)
- [counts.py](counts.py)
- [finalize.py](finalize.py)
- [preflight.py](preflight.py)
- [prepare.py](prepare.py)
- [run.py](run.py)
- [unit.py](unit.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L10](RESULT.md#L10)
- [REEXAMINATION_INDEX_20260919.md#L563](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L563)
- [REEXAMINATION_INDEX_20260919.md#L566](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L566)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-exact-zero-leaf-operands-save-less-than-0-00009) · [Copy provenance](../../../PROVENANCE.json)

