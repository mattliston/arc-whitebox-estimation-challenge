# Young-only negative-tail spherical trace, one fixed pilot

This is source preparation, not numerical authorization. Parent is official S33
source `88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6`.
Append only the pinned fragment. Gaussian3/E2/EL2, all learned literals, original
3.5 outer deadcut, old core/join/U/OTT reconstruction and trace remain unchanged.
The setup remains 682 bound, 649 existing fused plus nine batched symmetry sides.

Only the post-join young/newborn push at layers 7–14 may shrink its output row
roster. Within the original outer wx columns, omit only a < -2.5 AND not in the
actual padded alive mask. Round the retained count UP to 64; do nothing unless
at least one whole 64-row block is saved. Stable ordering keeps all protected
rows and retains the least-negative padding rows. Zero-expand into the original
outer order before concatenation with the unchanged old bank. Preserve previous,
newborn, lam, old state and full lower moments. Suspend any full-W W5 grant during
the cropped young call and restore it in finally. Lost grant benefits are a cost.

For stored young columns p_j and ACTUAL current signed coefficients lambda_j,
tau = sum_j lambda_j p_j ||p_j||². Inherited p is compact on previous and its
trace is scattered back into the full 1024-dimensional input space. Each newborn
pair is B_j+t_j e_selected and B_j+s_j e_selected, using stored BD base32 (zero
diagonal), t/s32 and the two actual coefficient slices, including S19 scales.
Its contribution is B_j*(ct+cs) plus e_selected*(t*ct+s*cs), with
ct=lambda_t*(||B||²+t²), cs=lambda_s*(||B||²+s²).

For each newly omitted column w of the full 1024-row weight matrix, restore
3*||w||²/(1024+2) * (w.T@tau) to its CP diagonal AFTER the original OTT repair.
This is the degree-one spherical projection of a homogeneous cubic, not exact
recovery of its actual diagonal or its conditional distribution. Full ambient
norms include weight rows outside previous. All source-dependent operations use
paid FNP, with F64 trace arithmetic. One length-1024 correction survives until
that layer's diagonal; entry/fallback/error/consumption/predict-finally clear it.
No model arrays are retained across prediction. No input mutation or new arena.

## Exact prior evidence and risk

The total-CP version was already implemented on S31: `candidate_dead_tensor_trace_v1.py`
(acc54b93), retaining only padded live rows, with old+young+newborn repair.
TRAIN2 raw +1.570598%, adjusted +0.378080%, bill -1.175998%, 1/2 wins; stopped.
Its conditional-mean follow-up `candidate_dead_tensor_conditional_mean_v2.py`
(8697a46e) was also tested: raw +1.92025%, adjusted +1.14522%, bill -0.76248%,
1/2 wins; stopped. Neither is an untested lead. This fixed pilot isolates only
young/newborn, a less aggressive negative cutoff, and leaves the old path intact.
No exact prior for that isolated scope was found in the bounded H2/D19 search.
These failures lower confidence; no positive endpoint forecast is asserted.

Old saved diagnostic row counts at 2.5 suggest 64–128 additional rows at selected
L7/10/14 sites, but are not current S33 counts or a saving certificate. Net work
must include F64 casts, source norms/trace products, full-weight projection,
row gather/zero expansion and any lost W5 grant. No old reconstruction credit.

## Frozen sequence and gates

Root alone executes the tiny helper fixture: NATIVE -B probe.py BUILD_SHA
primitive_result.json under 60 seconds / 1 GB / zero swap / BLAS1. It checks the
exact stored signed trace against explicitly assembled columns, harmonic scale,
full-vs-compact row indexing, padded-live preservation, 64-block fallback,
unchanged inputs, diagonal combination and exception/owner cleanup. It does not
claim predictor equality or empirical accuracy. No estimator import.

Then native TRAIN2 ranks 35,100 versus exact S33: strictly negative pooled
adjusted delta and both wins. All 16 outputs finite and L0–6 byte-exact to S33.
Guard 150 seconds / 7.5 GB / zero swap / BLAS1. If admitted, early boundary2 must
pass the existing separate S32 timing anchor: projected residual <=.35 seconds,
wall <=105 seconds, setup <=4.1 seconds and existing receipt/handle/cache gates.
Only then fixed TRAIN6 ranks 36,45,52,58,64,70, guard300 seconds, same resources.
Pooled TRAIN8 must gain >=.1% versus S33, win >=6/8, bootstrap CI upper <0, and
additional6 must improve. Use the frozen native engine 50k bootstrap seed20260926.
No raw veto; no cutoff/layer/rounding/trace-scale/direction/precision rescue.
Failure preserves receipts and stops this source. No validation admission here.

Any bank still requires separately admitted fixed VALID40 >=1% versus latest
official source, >=28 wins, CI upper <0, max network regression <=10%, followed by
complete serving qualification and root release. This pilot grants no release.
