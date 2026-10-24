# Per-layer Gaussian gate reuse on official parent 331684

Prospective 2026-09-20, before a candidate or new numerical run. This is a
runtime study; the approximately 1e-9 goal and previous failed accuracy gates
remain unchanged. Parent source SHA256 is
facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
The official bank is adjusted 3.5844144212297072e-9, 100/100 completed.
Its largest reported residual time is 0.396147558 seconds. Completion does
not establish spare capacity on every future run.

The current predictor computes maximum(v,1e-10), its square root, mu/sigma,
and the Gaussian CDF for selection. gaussian_raw repeats those operations
later in the same layer. Retain that four-array tuple in the current loop
frame and explicitly pass it through the pair mapper into gaussian_raw.
There is no network-dependent global cache or cross-layer reuse. The
predictor does not mutate mu or the preactivation covariance in between;
gaussian_raw retains its default original path when no tuple is supplied.
The final scalar shortcut retains that default path. All model parameters,
physical equations, precision, selected indices, operation associations,
existing matrix kernels and bookkeeping are unchanged.

Build from the exact parent by checked edits to gaussian_raw, three pair
mapper signatures/call sites, and the last active _predict_binary only.
The flag _GAUSSIAN_GATE_REUSE enables the change; a separately pinned
disabled source differs only in that flag. No new numerical work occurs
outside predict. Do not change Flopscope, timers, garbage collection,
allocation charging, dispatch, budgets, or the SDK timing convention.

Units: compare original/default/reused Gaussian tuples in FP32 and FP64,
including zero, negative and tiny input variance and extreme finite means.
Compare all outputs and auxiliary arrays of the three pair mappers on
finite positive-variance fixtures. Require byte equality, exact own repeat,
no input mutation, and a positive measured removal of the duplicated work.
Measure the 1024-vector removal independently and require the full native
bill decrease to equal 15 times that amount. The last scalar layer has no
duplicate selection gate. Preserve all default-path behavior.

Native gate: first one disabled cold35 equals the frozen parent all16 bytes
and full bill; then enabled35/100/35 and independent cold100. Compare against
already sealed exact current-parent controls in
full_lower_multilayer_mean_observer_v1/parent_triple and parent_cold100.
Require all16 bytes exact, finite/nonnegative arrays, reduced complete cold
and warm bills by the declared amount, own repeat35 and warm/cold100 exact,
and existing numerical caches retired. No truth arrays, fits or samples.

Only after native qualification, run five fresh ordinary NumPy-free
prediction-only timing pairs on ranks [35,100,37,39,42]. Alternate order
parent/candidate then candidate/parent, starting parent/candidate. Reuse
the existing SDK-scope client/driver: module load and Estimator construction
are outside the predict budget as in the installed SDK, all prediction work
and lazy initialization stay inside. Parent is exact331684, not an earlier
unpacked source. Do not profile or patch the accounting implementation.

Require all16 bytes exact in each pair, complete bill decrease exactly as
qualified natively, at least4/5 lower candidate residuals, summed candidate
residual / summed parent residual <=0.98, each ratio <=1.10, every candidate
residual <=0.25s, summed total-wall ratio <=1.05 and each <=1.10. These are
fixed local exploratory runtime gates, not a platform speedup claim. Failure
closes the study without repetitions, wider sampling, code tuning or upload.
If positive, one ordinary enabled35/100/35 checks exact native/client bills
and output bytes, own repeat/warm-cold behavior and residual <=0.25s.
Then the current SDK contract and exact source/archive validation are
required before upload under standing user authorization. Official grading
and 100-network completion determine banking; no duplicate existing upload.

Use one nice10 numerical group, six BLAS limits1, 8GB RSS, zero added swap,
and the existing 256MiB emergency swap stop. Units90s, each cold120s,
native triple180s, ordinary triple240s, controller1500s. At most24 full
estimator calls and 150MB new artifacts. No GPU, new dataset, label access,
fit, alternative precision, gate threshold or timing-sample rescue.
Sources and reused controls must be sealed before each stage executes.

Source review: post_covariance_runtime_source_review_v1/REVIEW.md identified
this duplicate before this study. The active source was checked directly:
vv is diag(Cpre), the selected subset operations read response_gate, and
intervening third-moment assembly does not modify mu or Cpre. The clipped
variance used by gaussian_raw remains distinct from the unmodified v in
each pair mapper's other equations. No earlier negative is reopened.
