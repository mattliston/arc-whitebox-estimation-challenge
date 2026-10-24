# One exact-shape serial-D6 composition on frozen P1

Prospective, September 9, 2026. Phase 2 only. This directory prepares one
source-only candidate. No numerical execution, fitting, target access,
shape/depth sweep, automatic launch or release is authorized by a build.
Root reviews the complete sources before explicitly running the builder.

## Fixed source and dispatch

Parent is P1/build_v1/estimator.py, SHA
0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435.
Preserve its complete learned P assignment, all heads, cstar, carrier,
gate_floor=.01, scalar precision, square schedules, 8192 tiling and all
unrelated source bytes. This does not compose the separate NODE experiment.

Modify only the LAST `_packed_product_4` for FP32 arguments of exact shapes
A=(1024,832), B=(832,8192), out=(1024,8192), with the original push view
owner `_SCRATCH[1]`. Dispatch these products to the frozen serial-top-D6
algorithm; every other call takes the original D4 return. No padding,
product copy, new output allocation, threshold change or alteration of the
frozen D6 algorithm is allowed.
The original `_packed_product_cached` remains byte-exact. All seven D5
children call it directly with levels=5, bypassing the D4 dispatcher.

Use kernel.py SHA
a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7
from current_d50_serial_d6_kernel_v1. Copy only its `_Binding` class and
`product` function, changing their otherwise-unused names reversibly.
Generate both maximum-capacity view-spec tuples OFFLINE in the stdlib
builder using that frozen module and its authenticated generic layout.
The candidate contains literal metadata, not a runtime layout generator or
an import of the kernel, NumPy, Torch, a truth file or the builder.

Insert one binding call inside the existing first-prediction scratch
initialization. Bind BOTH serial push/read view sets, even on a network
with no dispatch, to the SAME original 292225024-element FP32 arena
(1168900096 bytes). Preserve the old arena, view sets and `_SCRATCH` slots.
The new 68 public FNP reshapes cost exactly 278566656+282489344=561056000F,
paid once in that first prediction's complete participant context. The
serial owners span 472026112 and639320064 bytes, not additional backing
arrays. Read views are deliberately bound but never dispatched by this arm.
The old and new views overlap; all kernels execute serially. Shape-only
frame caches retain arena views, never caller inputs/outputs or CP banks.
Do not reset candidate caches between calls or manipulate the meter.

Remove the appended helper block and reverse the two AST-selected source
edits to recover every byte of P1. Also prove unchanged import statements,
GNN/cstar literal spans, cached-child helpers and all other definitions.
The null candidate is the exact parent file, with no extra binding work.

## Evidence and scope of the estimate

The old max-read experiment remains FAILED at its original1.35 timing gate:
result86ce7e8b1ea3d9dcc81c42bad6cb78828b520f38aed5f4918acf6052a6cf8660.
Its synthetic (1024,8192,1024) shape is not the selected push. Do not relabel
that failure, infer a global schedule, or claim that D6 preserves D4 bytes.

The new fixed push832 probe completed with result
bf79b160f8952db1f433093c245c5497d2d25a623db6888cafa13f0ae48aaeb2.
Serial and breadth-D6 output bytes and repeat bytes agreed; all actual
tariffs/bindings passed. D4 warm productF=8417538304; D6=7062260240,
saving1355278064F per dispatch. FP64-reference RMS errors were6.7533e-6
and2.7611e-5 respectively: these are kernel errors, not neuron mean errors.
The measured positive setup difference=.003252666s and worst-serial minus
best-D4 warm difference=.521406291s gave3.131690412s for six occurrences.
The passing <=8s criterion admitted a partial-shape whole-card test ONLY.

Saved P1 native40 result
7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3
contains all47 selector counts. For push layer l, contracted width is the
previous active count a[l-1], and inherited columns are
R_old=2*sum(b[j],j=0..l-2). The newest birth is a separate <=1024-column
common product. Exact selected hits are
1[a[l-1]==832]*floor(R_old/8192), l=1..15. All widths are32-aligned;
full8192 tiles need no padding and take the D4 rectangular path.
This reconstruction gives172 hits across40 primary networks, mean4.3,
maximum9 (rank149), and no hits on130/204. Repeat37 has2 and is excluded.
Independent d50 rank35 selector metadata reproduces the six recorded
census tiles exactly. The builder repeats these metadata-only checks.

With all other work kept at its actual saved bill, and conservatively
charging561056000F on EVERY primary network, meanF would decrease from
468321676751.725 to463055037076.525:1.124577387%. The actual single-process
once-only setup convention would give1.241383768% on that same frozen
trajectory. These are COST ESTIMATES, not candidate bills or scores.
Changed FP32 arithmetic can change future selectors, counts and raw error.
The nine-hit timing heuristic is4.695909285s; it is neither a whole-card
nor official-worker runtime certificate. This is a small Pareto-component
opportunity, not a milestone solution or an asserted raw improvement.

## Later numerical and endpoint gates

No numerical harness is created here. A separately reviewed launch must
acquire the fixed existing VALID40 plus repeat37, using the qualified
official-owned FP32 input/predict/materialize/closed-bill/public-reset loop.
Use the same candidate for raw error AND its own complete measured bills.
Compare to authenticated P1 native40 controls, matching cold37 and warm
other primaries; do not transplant an estimated or parent bill. Repeats
are excluded. Observe actual dispatch counts and all paid view bindings;
prove cold/repeat initialization from measured nonoverlapping regions.
Do not require old zero-F operation counts across changed cache histories.
Require all16 finite prevariances>=1e-10 without flooring, all47 valid selectors,
finite nonnegative served outputs, exact repeated bytes, released contexts,
own F<=2^41, and literal/source identity. D4 output-byte parity is not a
new requirement: different rounded arithmetic is the explicit intervention.

Seal all outputs, inputs, bills and sources before VALID truths. Report raw,
mean completeF, and mean of per-net raw*max(.1,ownF/2^41), not a product of
population means. Advance only with strict mean adjusted improvement over
P1 AND at least24/40 adjusted wins. Report Pareto only if mean raw and meanF
both do not worsen and at least one improves; otherwise an adjusted gain
is honestly a beneficial tradeoff. No fit, scaling, tolerance rescue,
alternative shape or second schedule is authorized by a failed endpoint.
Next require strict NC-adjusted improvement over P1 on BOTH independent
public8 and virgin6 with their own complete candidate bills. Keep all
previous failures and distinguish small components from milestone results.

Numerical launch requires an empty numerical/MPS slot and one fresh nice10
OFFPY child, CPU BLAS1, no fallback, bytecode disabled. External monitoring
must include the supervisor and its entire owned process tree: RSS<8GB,
zero added sampled system swap, samples at most1s apart, every context<120s,
no completed bill+released-lifecycle interval>=120s, whole native40<1900s.
The source-only builder has60s/2GB guards and exclusive outputs. A later
saved-output score has120s/1GB guards. Sampling is not a continuous bound.
Kill/reap only owned groups, preserve failures, and do not overwrite files.

Unchanged production qualification still requires five persistent NumPy-free
contexts with total<60s and residual<.4s each, combined client/server<8GB,
complete F<=2^41, exact repeats, native numerical fidelity and all package
checks. No upload except a separately preauthorized validated milestone;
no forum posts, commits, promotion, frozen edits or root-estimator changes.

## Builder interface

Inert until `build.py --build --source-sha256 HASH --protocol-sha256 HASH`,
with bytecode disabled (`PYTHONDONTWRITEBYTECODE=1` or Python `-B`).
Exclusive build_v1 holds estimator.py, null_estimator.py, result.json and
completion.json. `verify_build(build_sha,candidate_sha,builder_sha,protocol_sha)`
authenticates source-only evidence and reruns the lexical construction; it
returns flat source_bindings/input_bindings and candidates parent/null/terminal
as {path,sha256}. No numerical code is imported or evaluated by this API.
The receipt exposes the two extra layout-literal names and paid tariffs so
an existing native observer can recognize them without adding arithmetic.
