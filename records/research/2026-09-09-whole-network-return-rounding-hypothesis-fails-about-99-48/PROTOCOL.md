# Fixed full-network return capture, v1

Frozen prospectively 2026-09-09 UTC, after source review and before any
numerical import or experiment. This document supersedes DESIGN_DRAFT.md
only as this run's executable contract; the draft and all old receipts remain.
Phase 2 only. No optimizer, fitting, target arrays, new direction/amplitude,
candidate edits, score estimation or derivative qualification is authorized
by this diagnostic. Official bank #330390 remains 4.53886349950395e-9.

## Decision and expected magnitude

Route A: numerically understood current-bank learning. The completed L1
return-quantization diagnostic (analysis result SHA256
759bc9902d388c6ad223742923955884294af4650dc76f083f4959614c1d4a39)
attributes 99.78–99.98% of signed native-versus-native-G0-CPU discrepancy
to final FP32 return rounding on the four original cases. That does not
diagnose original CO whole-network outputs, nor qualify gradients. Expected
score gain is UNKNOWN: 4e-9 and 3e-9 require 11.9% and 33.9% raw reductions
at current cost. With a 0.1 multiplier even 1e-9 requires >52.8% raw reduction.
This is an information acquisition with direct new numerical evidence,
not an assertion that the failed ND2 fit or old derivative gate has passed.

Route B: structured variance supply. Old D11/HV rank35 half-truth variance
improved raw about 24%, not on the current d50 consumer. Current-bank
oracle value, authenticated independent reference, causal supplier accuracy
and complete deployable cost are missing. No new head/MC run is justified
by that old number. Source audit is recorded separately in
VARIANCE_SUPPLY_PROVENANCE_20260909T1004Z.md; closed families stay closed.

Select A's cheapest decisive full-return capture, not fitting. Deployment
cost is unchanged because no deployment modification is made. Planning
experiment cost: ten native calls about 4.71T total metered FLOPs, plus ten
unmetered offline CPU forwards. These are estimates, not scores or new
meter receipts. Prior native calls about 20s and twelve CPU forwards about
148s suggest 5–7min; this run has an independent 900s hard cap.

## Exact roster and frozen data

Twenty serial fresh-child forwards: native first, then CPU; each role in
case order null, plus025, minus025, plus050, minus050 and rank order 35,100.
Use already-built BR candidates and already-sealed CO payloads/cotangents.
Cases correspond exactly to original h=0.025,0.05 and their negatives.
No regenerated probes, rank selection, target access or hyperparameters.
The launch manifest pins all six source/protocol files and the inherited
stdlib watchdog helper. Its digest is supplied independently at launch.
Each child authenticates original receipts, P/non-P/cstar, both canonical
weight files and source ownership before numerical imports. Recheck all
observed bindings before completion. Do not inspect/hash truth arrays.

CPU uses unmodified original CO.PCard with training=False, no native-G0
injection, no MPS, no gradient. One existing FP64 prediction return is
saved before serving. Require all original served FP32 bytes and original
CPU64 synthetic scalar to match exactly, all 16 original unfloored variance
diagnostics and 47 ordered selectors byte-exact, and every P/cstar byte
unchanged. No parent/null rerun beyond the fixed twenty-cell roster.

Native uses exact original candidate source, official independently copied
FNP FP32 input arrays and actual estimator seed. A passive sys.setprofile
observer keys on exact active function code objects. Gaussian v is exported
at call entry before its floor; selector ids/mask at return; only the
existing sixteen FP64 means are retained at _predict_binary return. No
frame, locals dictionary, covariance, weights or CP bank is retained.
Export existing-array bytes inside the paid prediction/materialisation
context, then clear references and remove the observer. No new FNP
arithmetic or participant state mutation. Save original raw, clipped raw
and served outputs. Require exact original BR served bytes and exact
FP32(clipped raw) reconstruction. Native selector order/membership versus
CPU is reported, not assumed; native variance hashes are not required to
equal CPU hashes. Native no-floor domains and full selector roster must pass.

All ten native calls own FULLCOLD bills, including initialization. Input
copy cost is recorded separately according to original grader ownership.
Preserve actual bills before fail-stop. Use original paid arithmetic block
and closed-context bill/reset/release ordering with executable AST checks.
No initialization subtraction or old warm100-plus-estimated-init surrogate.
New cold35 bill must match original cold35 exactly; cold100 is measured
fresh without requiring equality to its old warm bill. Actual observer
wall time and complete FLOPs are reported.

Repeat contract: no additional repeat calls in this causal diagnostic.
Each fresh served cell must equal its original endpoint; the original
BR35/100/35 records and exact repeated35 bytes are authenticated. This is
not a new persistent-estimator lifetime/repeat qualification. Any future
release still needs the complete existing repeat/residual/resource gates.

## Judgment fixed before execution

Use original CPU64 linear functional and cotangents: last-layer inner
product divided by sqrt(1024), plus 0.05 times all earlier-layer inner
products divided by sqrt(15360), preserving Torch reduction grouping.
Reproduce original endpoint scalars and symmetric scalar responses.
Let D_y be native-minus-CPU symmetric response of served FP32 endpoints
at each h/rank; D_z the same response of nonnegative FP64 raw returns,
without FP32 recasting. Q=D_y-D_z. Report all four cases, signed scalar
and vector discrepancies, exact decomposition checks, and all sixteen
layer contributions. Clipping is explicit, never applied to differences.

The causal wording 'return rounding dominates' requires ALL four cases
to have norm(D_y)>1e-10, abs(S(D_y))>1e-10, norm(D_z)/norm(D_y)<=0.25 and
abs(S(D_z))/abs(S(D_y))<=0.25. Near-zero denominators are INCONCLUSIVE,
not passes. Publish absolute quantities and all failed cases. Numerical
completion PASS is distinct from this scientific classification.

Original whole-network 5% served-response gates stay FAILED regardless of
the result. No auto-waiver, derivative qualification, optimization admission
or score gain follows. Any follow-on needs a distinct prospective contract
covering rounding, numeric fidelity, raw accuracy and complete cost.

## Resources, provenance and failure

Only one numerical child at a time, nice10, all five BLAS controls=1;
CPU Torch threads/inter-op=1, no AP threadpoolctl, no MPS/fallback.
Native fixed OFFPY NumPy2.4.6/flopscope0.12.0/whestbench0.16.0;
CPU/judge AP NumPy1.26.4/Torch2.7.0. Source helpers import inertly.
Native candidate may import only its existing permitted participant stack.

Each full child <120s and lifetime RSS <8,000,000,000 bytes. Supervisor
plus one child <8,000,000,000 bytes sampled at intervals <=1s; whole900s;
system swap growth <=512MiB. RSS/swap sampling is not a continuous
guarantee, and swap is system-wide. Supervisor may terminate only its
recorded fresh child process group. Reap each child and require no surviving
descendants. No retry, threshold relaxation or extra numerical call on failure.
Judge uses saved outputs only; no network, weights decoding or targets.
Watchdog includes judge in the same global wall/resource budget.

Create outputs exclusively; immutable result/completion SHA seals at every
stage. Preserve failures, attempted bills and historical receipts. No board
submission arises from a capture result. The already-requested confirmation
bank #330390 is complete and must not be duplicated.
