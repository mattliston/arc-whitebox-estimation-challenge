# Closed raw guardrails versus the adjusted objective: receipt audit

September 6, 2026 PDT. Read-only source/JSON-receipt audit authorized by
root. No estimator execution, target-array access, refit, port, benchmark,
threshold/schedule change, or edit to any frozen source or closeout.

## Bottom line

**The two closeouts do not establish adjusted-score worsening.** They
establish failure of their preregistered raw-worsening guardrail.

There is one genuine same-frame favorable adjusted comparison already in
the archive: **unlearned response-row reduction on rank35, old native D5
implementation, improves raw times fully metered estimator FLOPs by
4.4730%.** That is not the later learned trajectory, the current D4 bank,
a 14-net result, or a persistent-runtime qualification.

Cold-output trace has no complete metered estimator receipt. Neither
family has an archived current-bank learned adjusted result below4e-9.
The fixed closeouts and their no-rerun rules remain in force.

## 1. Exact old native response-row comparison

These two archived receipts measure output error and the complete
`Estimator.predict` BudgetContext in the same run, on rank35 versus the
same stored1e9 truth:

| Old unlearned native arm | Raw final MSE | FLOPs | Utilization | Raw times utilization |
| --- | ---: | ---: | ---: | ---: |
| Parent | 2.4645930599236925e-8 | 450565504587 | .20489346961176125 | 5.0497902322883275e-9 |
| Response rows | 2.5006931178899064e-8 | 424198161779 | .19290299031990799 | 4.823911803133771e-9 |

Raw ratio1.01464747205, cost ratio.941479446297, adjusted ratio
**.955269740174**. Both utilizations exceed the0.1 scoring floor.

Receipts `response_rows_parent_cost_r35_v1.json` and
`response_rows_cost_r35_v2.json` have actual mtimes September4,
22:52:54 and22:54:29 PDT. `build_response_rows_cost_v1.py` derives both
controls from `candidate_fullpair_runtime_d5_t8192_v1.py` and explicitly
applies **no learned correction**. V2 fixes scalar immutability only.
`screen_radial_fnp.py` measures the complete predict call but allows900s
locally and contains no production residual-time gate. Thus these are
complete native arithmetic costs, not a proof of production reliability.
No corresponding complete native rank100 cost receipt was found.

The old NumPy unlearned raw ratios are slightly different
(rank35+1.5130%, rank100-4.0920%). Do not substitute those raw values into
the native cost receipts and label the result measured.

## 2. Learned response-row guardrail was not an adjusted test

`response_rows_closedloop_r35_v1.json` and its rank100 counterpart use
`payload_response_rows_v1.npz` through `screen_radial_closedloop.py`.
The matched parents are `compact_feed_fullpair_closedloop_r35_v1.json`
and rank100. This is NumPy, full-final-layer features and the learned
per-layer head, not the bank's later scalar finalQ readout. No complete
metered learned candidate, on-policy scalar-final refit, or candidate
parity was archived.

| Rank | Learned raw ratio | Complete cost condition for adjusted improvement |
| --- | ---: | --- |
| 35 | 1.031203959728 | Cost ratio <.969740263860: save >3.02597% |
| 100 | .996477285140 | Cost ratio <1.003535168250: up to.35352% cost growth |

Conditionally applying the OLD unlearned native cost ratio.941479446297
would yield learned adjusted ratios.970857333024 and.938162882661
(-2.9143% and-6.1837%). **These are counterfactual arithmetic, not measured
learned scores:** masks, feedback, final-layer work, precision and head
ownership have not been jointly validated. They show why crossing the
3% raw guardrail by.1204 percentage points is not evidence of adjusted
worsening. They do not authorize changing that gate after seeing results.

## 3. Cold trace: necessary savings, but no complete cost

From `cold_cp_trace_r35_v1.json` / rank100:

| Rank | Unlearned raw ratio | Cost saving needed for adjusted improvement |
| --- | ---: | ---: |
| 35 | 1.032364486856 | >3.13498646% |
| 100 | 1.003996328402 | >.39804213% |

The14.1653% /14.0655% archived figures are interior CP row-count proxies,
not whole-card savings. `radial_cold_cp_trace_v1.py` adds all of:
FP64 factor-norm reduction and trace-vector multiplication,24 full
covariance-vector iteration steps, a covariance Frobenius norm, another
weight-vector contraction, and new zero/scatter assembly. None is free.

Moreover, the source changes the final-layer cold scalar reads too, while
`states` is appended only before the final layer. Its receipt therefore
does not even record the final alive geometry needed to reconstruct all
changed products. The old NumPy reference also uses full-width products
where the current bank already uses active input compaction and shares the
youngest pair. The proxy cannot be transferred directly to the D4 bill.

## 4. Current-D4 shape-only opportunity accounting, not a score

Using only archived JSON metadata, the stored interior active/birth/column
counts were cross-checked against `uniform_half_cp_shape_cost_v1.json`'s
original current-bank geometry. Its original plane prices matched exactly.
No masks, state arrays or target arrays were read.

For the affected large products, exact stored D4 polynomial:

    M(m,k,n) = 1.17236328125*m*k*n
             +19.55078125*(m*k+k*n)+10.171875*m*n.

All nonempty bulk products here have a dimension>1024 and dimensions
divisible by32; no square-depth choice or padding approximation is used.
Apply the current8192 tiling, charging output concatenation m*n for a
wide output and one m*n addition per extra inner-dimension tile.

For cold trace, sum only the interior inherited-column push difference
M(1024,a,Rold)-M(c,a,Rold), layers1..14. Exclude youngest-pair and final
changes entirely. For response rows sum
M(c,R,c)-M(response_rows,R,c), layers1..14. All other costs remain unpriced
in these subtotals.

| Current geometry | Rank35 | Rank100 |
| --- | ---: | ---: |
| Archived D4 bank first-context FLOPs | 490645310609 | 489553809753 |
| Cold interior inherited-push gross reduction | 26744438676 | 26471263392 |
| As fraction of bank | 5.45087% | 5.40722% |
| Response interior plane gross reduction | 32668253840 | 34261053904 |
| As fraction of bank | 6.65822% | 6.99842% |

These are exact arithmetic projections for the specified **partial
blocks at frozen geometry**, not complete-cost bounds on a changed
learned trajectory. Cold excludes all trace/solve work and final geometry;
response excludes its FP64 radial-restoration contractions, selection and
changed elementwise/scatter costs. Both exclude feedback-driven shape
changes and any new final-head work. They support an opportunity audit,
not a claimed adjusted win. Their single-digit scale does not establish
the approximately19.7% saving needed to move the current official bank
below4e-9 at unchanged raw.

Inputs: `direct_kernel_schedule_prices_v1.json`,
`bank_matmul_census_v1.json`, `uniform_half_cp_shape_cost_v1.json`,
`fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json`, and the four
cold/response unlearned JSON receipts. Current bank source SHA remains
`3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757`.

## Authority and interpretation

The mathematically relevant comparison is raw ratio times scoring
multiplier ratio, not a raw-only cap. The earlier cap was a conservative
preregistered research stopping rule. This audit distinguishes those facts
without rewriting history or declaring the closed family open.

Under the standing prohibition on computation inside a closed family,
reopening either exact round or changing its gate after these results
requires explicit user authorization for that exception and a new
prospective scope. Root's present audit request explicitly grants neither.
No refit, port, threshold/schedule variation, benchmark, panel, or upload
follows from this note.
