# Positive-tail trace and 600 setup keys: held-out adjusted score improves 1.73538%

2026-09-26 · experiment · adopted · measured

The selected cutoff-2.5 repair was combined with 600 keys selected only from training shape incidence. The standalone bundle stayed below the then-current submission threshold and later supplied the two-tail submission.

Recorded measurement:

    Training shape coverage 56.88% ->98.08%; all624 setup entries bind; TRAIN2 adjusted -2.29806%; fresh VALID10 adjusted -1.73538%, paired95% CI [-2.16259%, -1.32768%], 10/10 wins; raw -0.16028%, CI [-0.52760%, +0.21765%]; bill -1.59069%; CPU -1.92825%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_old_tail_trace_radial_z25_v2_prebind600.py](code/candidate_old_tail_trace_radial_z25_v2_prebind600.py)

## Evidence

- [RESULTS.md#L469](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L469)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [trace_bundle_selection_protocol_v1.json](trace_bundle_selection_protocol_v1.json) · result

</details>

## Related work

- shipped in: [Submission #332517: two-tail old-row reconstruction with analytic diagonal and radial restoration](../../../submissions/332517/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-positive-tail-trace-600-setup-keys-held-out-adjusted) · [Copy provenance](../../../PROVENANCE.json)

