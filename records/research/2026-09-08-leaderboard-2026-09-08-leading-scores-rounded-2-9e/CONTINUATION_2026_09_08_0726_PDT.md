# Phase 2 continuation — September 8, 2026, 07:26 PDT

Work is continuing, not paused or completed. Official target remains strictly
below 1e-9, with qualified milestone submissions below 4e-9, 3e-9 and 2e-9.
No forum posts, commits, promotion or frozen-work changes. Two existing agents
only; at most six nice-10 numerical workers and one MPS job. No numeric process
is currently live. Last measured swap: 776.75 MiB; free disk about 529 GB.

## Bank and score evidence

Official #329926 remains adjusted 4.982231460183304e-9,
raw 2.2448352261505987e-8. Source
`candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py`, SHA256
3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757.

Best qualified local exact-output component remains
`unused_materializations_v1/estimator.py`, SHA256
409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f:
14-net independent-4M unclipped NC raw 2.2458484596872894e-8,
complete mean cost 466058731667.9286 FLOPs, adjusted
4.770235434670105e-9. Public8 adjusted 5.183911640883259e-9,
virgin6 4.218667159719234e-9. Init charged once and repeat excluded.
Remote/runtime/package checks preserved. Not a milestone; no upload.

Primary leaderboard checked this turn at
https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards
showed two leaders rounded to 2.8e-9, and ours rounded 5.0e-9/rank7.
This is a dated live-board observation, not a new local or official result.

## Frozen native-point certificate failed; independent diagnosis passed

`current_response_rounding_v1/qualification/result.json`, SHA256
3cf6986864a238c4c71a68287e385006d339b3c276703ac2d995281edab12a7d,
remains FAILED. Its seven rank35 contexts passed. Rank100/layer8 passed
all native/lifted byte checks, boundaries, every second derivative and GNN
derivative check. Only feature21 first derivative at the original coarse
step .001 sigma failed; all later rank100 contexts were unexecuted.
No truths were opened. The earlier whole-FP64 convention and its feature27
value mismatch remain closed and unchanged too.

One independently reviewed, prospectively sealed scalar diagnosis was run
ONCE, using the SAME four units (0,257,514,771), both existing step sizes,
and the already captured rank100/layer8 state. No model, step, node,
coefficient, tolerance, fit, covariance action or head was changed.

New files in `current_response_scalar_certificate_v1/`:

- `scalar_interval.py`: 80a2cb4ec800648a4c7792fd019f9154f29d43b9dbc8d559b15d0329edc520b0
- `diagnose_scalar.py`: 799ae77214f8fd4cfdfe1733a0e2becc2e69bd2da589cfbecfc223e6826469f9
- `PROTOCOL.md`: 2f19f4765c8b9102161384e0ab3f8b73a75bc22df38441e70a1d27e9b1b8af8b
- `diagnosis/result.json`: 14b3dbabcd594587f9898acb1f8e3287932e00e9e25bf17347ff5c14a2d92227

Diagnosis PASS, terminal exit0; completed 14:23:51.961 UTC /07:23:51 PDT,
wall0.4198538754s, peak141541376 bytes. Original norm discrepancies
reproduced at both steps. Outward independent-reference AD norm errors:
first3.142626e-13, second3.425700e-13, both below fixed1e-9.
The ideal coarse first-FD truncation norm is2.3087118266e-7, versus the
original allowance1.2365401150e-7; floating FD evaluation error is
3.0265074214e-10 (coarse),4.2340090570e-10 (half). Even subtracting both
floating FD error and AD error, the lower bias bound exceeds the old guard.
The coarse reference is thus demonstrably insufficient. This diagnoses
the failed reference; it does NOT turn the old certificate into a PASS.

The independent implementation uses private directed Decimal intervals,
fixed precision160, certified Machin/CDF tails, and an independent
polynomial-CDF derivative recurrence. It retains both floor/sigma and
sigma/raw-v ratios and exact-real sigma^4 / actual binary denominator.
No adaptive precision, scipy, mpmath or original Jet differentiation.
The detected threadpool list was empty: BLAS environment guards, not an
observed pool measurement. Exact OFFPY versions and nice10 were checked.

## Current next work and gates

Accuracy agent is authoring an INERT additive successor certificate under
`current_response_certified_native_v1/`. Conditional on the sealed diagnosis,
uniformly replace feature21's inaccurate FD acceptance oracle with independent
interval-reference first AND second derivative error <=1e-9 at ALL original
derivative contexts. Preserve its literal FD diagnostics, every other old
criterion, all14 current-native/head contexts, source seals, and resources.
Nothing in that successor has been admitted/executed yet. Primary/full
independent source review and a new admission are still required.

Only a passing successor can admit the fixed no-fit current response gate:
rank35/100, all seven layers8..14 equally, independent2^20 A/B gate truths,
learned cross-MSE <=.90 host AND <=.80 Gaussian on BOTH nets, positive
denominators, noisy-stream and paired batch-jackknife reporting. This is
information evidence, not final MSE or adjusted score. The original inert
scorer is under `current_response_rounding_v1/score_gate_information.py`;
do not run it against the failed old certificate.

Only after that gate passes, qualify the independent covariance algebra on
a tiny fixed synthetic fixture and consider the fixed no-fit rank35/post12
quadratic covariance action at amplitude1. Keep captured P correction,
centering, masks, nextW and other moments fixed. Require next-mean cross-risk
<=.75 current and EACH noisy stream <=.85, plus numerical/domain checks.
No curvature, layer or strength rescue. No gate/covariance truth opened yet.

The eventual substantial advance criterion is actual final raw times OWN
complete cost <=.90 parent on BOTH public8 and virgin6; raw alone does not
decide. A +8% complete-cost ceiling is a rejection gate, not a measured bill.
Before integration, account for features/two GNN jets, covariance action,
all downstream consumers, state births and runtime. Efficiency agent is
doing a source-only additional-cost audit now; no complete estimate exists
yet. Small independently validated components stay preserved.

The two strongest remaining routes are (1) this raw-response/covariance route,
supported only by older first-response information gains, with current raw
benefit unknown; and (2) complete-cost reduction, for which no new material
faithful circuit has a credible endpoint-error and full-cost witness. At
current raw, uniform savings must exceed16.1467% for4e-9 and37.1100% for3e-9;
the .1 scoring floor prevents cost-only1e-9. No substantial run is justified
for the latter. Do not reopen the closed family catalog or the already-used
one-shot response-row exception. The earlier 04:51 continuation and frozen
closeout receipts retain the full previous catalog.
