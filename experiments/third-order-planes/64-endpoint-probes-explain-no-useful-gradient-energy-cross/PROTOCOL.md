# Fixed endpoint-pair span information test

Prospective, September 9, 2026. Source implementation only until root reviews
and explicitly launches the worker and independent supervisor. This is not a
candidate, compression, parameter fit, score, or native-gradient certificate.

The scientific specification is DESIGN.md SHA256
`461713beeb82d47299d2f0c02e78bed2598d6f675eaee38e1a74062d09d3c839`.
The fixed parent is P1 native source
`0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435`,
with its sole completed full FP32 P payload; frozen BD coefficients and all
non-P fields remain unchanged. Targets, candidate imports, flopscope imports,
network/subprocess calls, MPS use, and parameter updates are forbidden in the
teacher. Only weights, parameter payloads, source and metadata, and newly
produced target-free diagnostic arrays may be read.

## Workload and deployment features

Fixed TRAIN ranks35,36,38,100, in that order. Each rank has one full16 ordinary
CPU reference forward, retaining a private constant post7 state, followed by
two suffix8..15 VJPs at relative old-pair coefficients c=1. This is exactly
four reference forwards and eight suffix VJPs, twelve model-call retirement
markers. The private prefix comes from these new reference calls, never from
a saved teacher fixture. Checkpoint replays are additionally counted.

The exact CO CPU PCard recurrence receives the sealed P1 payload explicitly:
CPU FP64 moment state, original FP32 CP products, all16 P heads and47 selectors.
The reference observer returns the original step result unchanged and keeps
only detached references to post7 state and ordered birth indices. No prefix
autograd graph is retained. A suffix VJP has one FP64 coefficient leaf per
old intact pair from cohorts0..6. Newborn7, all factor bytes, C/m/b/radial/H,
and every original P tensor are fixed at the cut. There is no coefficient
change, physical column removal, or non-null card evaluation.

Both scalar weights of each old pair depend on the same relative coefficient.
The live repeated-plane restoration also depends on that coefficient. Use
the frozen whole_pair_kernel_core_v1.py `_gpk_plane` body with only explicit
NumPy-to-Torch spellings and functional diagonal assignment. Its source
inverse must recover the exact original function. Read old and live planes
using actual stored FP32 factors/weights and separate FP64 cubic diagonals.
The additive star companion is implemented as

    delta = LIVE_plane - OLD_plane
    L(delta) = 3*delta.T with diagonal replaced by diag(delta)
    star_live = cast32(star_original64 - scatter_active(L(delta)))

This is the original additive OLD-LIVE repair, written with subtraction so
adding positive zero cannot erase an original negative-zero star entry at
the null. It is not a detached/straight-through repair. All nine cut-state
tensor bytes must equal the reference at c=1. The suffix raw and served means,
prevariance records and ordered selectors must match that same CPU reference.
All later state paths and checkpoint recomputations remain live and audited.

The fixed64 deployment probes are32 Walsh mixtures of the existing H7 and32
weights-only backward probes W8...W15 R, with masking, FP64 normalization and
FP32 storage exactly as DESIGN.md. Future gates/states/providers are not
acquired. The frozen `_gpk_features` source is executed verbatim with NumPy
as its array namespace; the module itself is not imported. Five FP64 thin
products produce the collision-subtracted intact-pair cubic features Phi.
Their exact real-arithmetic formula and source owner are recorded. Numerical
native equality for this prospective feature constructor is not claimed.

The two dyadic +/-1/32 final-output cotangents and the pair folds are generated
from the exact SHA256 tags in DESIGN.md. The original final nonnegative FP32
served operand is reduced in FP64. Its AD is a smooth-lift information
teacher, not a derivative of the discrete native return map.

## Fixed analysis and verdict

Standardize the64 feature columns by their uncentered RMS over all pairs;
zero columns remain zero. For each of two cotangents on each rank, fit on one
fixed pair fold with ridge1e-6 and predict the other; then reverse folds.
There are sixteen small64-coefficient diagnostic solves. No intercept,
alternative features, adaptive rank/ridge, pseudoinverse, seed/step sweep or
P/coefficient optimizer is allowed. Per-cot/per-network coefficients are
teacher-only capacity diagnostics, not a deployable learned metric.

Operational `passed` requires source/input integrity, all null/domain/replay
checks, finite features/gradients/solves, the exact4+8 roster and retirement.
`information_promising` separately requires nonzero teacher-gradient energy,
per-rank aggregate cross-fitted residual-energy ratio<=0.10 and every
individual cotangent ratio<=0.20. Zero-energy cases are inconclusive, not
vacuous passes. A scientific failure can be operationally completed. Two
cotangents per network are not a full-operator confidence certificate.

No field may claim measured inference FLOPs, native raw accuracy, an adjusted
score, a deployable predictor, or a production qualification. A promising
span does not bound a finite replacement's nonlinear/membership/conditioning
error or its signed cross-term with bank error. Any later candidate needs a
new source/cost protocol and actual same-configuration raw_i*max(.1,F_i/2^41).

## Accounting scope

The original whole-pair kernel failure remains failed: old two-net complete
F474.949B ->384.590B, raw6.064x worse. Its mean64.921B cross/selected-Gram
subtotal could hypothetically be replaced by roughly12B cheap construction
while retaining all old restoration/other overhead, giving roughly331.67B
on that OLD host. That approximately30% estimate is not a current P1 price.

This worker reports only classical feature-GEMM subtotals: five FP64 products
20*64*n*M, plus eight FP32 future-weight products and one FP32 hidden product.
The complete constructor still must price powers/casts/gathers/guards,
selection/reweighting, small solves, packing and binding. Current frozen-state
remaining push/read tariffs,8192 chunks, square-depth binding, omitted source,
full diagonals, changed geometry and complete native wall/RSS remain unmeasured.
There is no teacher/provider/adjoint acquisition hidden in a deployment bill.

## Execution, receipts and limits

One fresh nice10 AP CPU process, NumPy1.26.4/Torch2.7.0; five BLAS launch
variables1, Torch intra/inter-op threads1, bytecode off, no threadpoolctl,
no MPS and no competing numerics. Cooperative child limits900s and12GB RSS;
independent watch imposes900s whole,12GB combined RSS, zero added swap,
120s without an ordered retired model call and actual completed observations
at most1s apart. No retries, deadline extensions or source changes after
launch. The unchanged native production contract is separate.

CLI: teacher.py --run --source-sha256 REVIEWED --protocol-sha256 REVIEWED
--watcher-sha256 REVIEWED. The watcher source is watch.py in this directory.
The output information_v1 must not exist. Runtime arrays use exclusive files.
Each atomic call_NN_complete.json and stdout event='retired' is published only
after checks and graph retirement. Calls are reference,cot0,cot1 for each
rank in fixed order. A constant cut/card may remain between those calls;
this is explicitly distinct from retaining a graph. It is retired before
moving to the next rank. Feature/analysis time is inside the same whole cap.

result.json contains passed/completed/hashes_unchanged/sources_unchanged,
reference_calls4,suffix_vjp_calls8,retired_call_count12, information_promising,
flat source/input/weight/artifact bindings, all rank diagnostics, and no-target
flags. completion.json binds result SHA. Mandatory finalization.json binds
both result and completion SHAs and records elapsed_s<900 after their I/O and
hashes. A late/missing/failing seal does not qualify an earlier PASS. Source
and weight files are rehashed before successful sealing; no target maps are
followed. Failures are preserved and all success flags fail closed.
