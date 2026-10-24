# Active covariance congruence512: fixed prospective component test

Phase2 only, September7,2026 PDT after22:10. This is a small numerical
compiler component, NOT a promised milestone or a new raw-accuracy source.
Qualified parent resident_gram512_v1/estimator.py SHA256
0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc.
Independent14 NCraw2.2458484596872894e-8, complete mean466705080466.9286F,
adjusted4.7768418485455745e-9. Official329926 remains4.982231460183304e-9.
Goal OFFICIAL<1e-9 and milestone banks<4/3/2e-9 remain unchanged.

## Two routes, evidence, complete cost and selection

1. Symmetrize the stored FP32 active covariance, then compute its final
   congruence product with the existing resident512 helper. The unchanged
   first product is Sym(C_AA)*W_A. Three512-output block products replace
   the final full product; four block copies mirror the cross block.
   Exact scalar source pricing after2*a^2 symmetrization and1024^2 copies
   saves239634432F per a1024 call,232336736 at992,225034944 at960,
   210419072 at896,195786816 at832,181138176 at768. All these products
   use the parent's fixed depth5 square schedule with no padding. This
   is NOT a D5 rectangular/runtime retry: rectangularCP schedules and
   arena geometry stay fixed. Fifteen calls in this width range imply
   2.717..3.595B reduction, a conditional complete-card estimate
   463.1106..463.9880B retaining ALL other466.7051B-parent work.
   This.58..77% estimate assumes unchanged downstream geometry and is
   not a measured bill. No exact current active-shape census was taken.
   Expected raw change is negligible only if the changed roundoff and
   explicit symmetry projection remain harmless; this is unproved.
   The existing initial/cold Gram passed complete14/remote qualification,
   providing implementation evidence for the helper, not proof for this
   new active product. Extra temporary storage is O(a^2), a few MiB;
   complete process RSS/residual/wall remain independent gates.
   Cheapest decisive test: fixed synthetic numerical/function-cost
   comparison below, followed only on PASS by fixed whole-card development.

2. An exactly centered four-region affine control supplies nonlinear odd
   information absent from shared affine controls. The new source audit
   REGIONAL_AFFINE_AND_LAZY_CUBIC_PREFLIGHT_2026_09_07_V1.md gives an
   eps^3/(3*pi) normal-form residual bound, not a random-network bound.
   Actual transverse/within-region residual variance is missing. The
   four-Jacobian128.85B setup and17-matvec35.65M/sample figures are leading
   subtotals, not complete bills; even ignoring setup needs residual
   variance<6.1681e-5 for IID adjusted<1e-9. No credible magnitude at1024x16
   is available. No sampling, fit, normal/rank sweep or candidate is
   admitted from that identity alone. Maintain this raw-information need.

Select route1's bounded numerical compiler qualification. Neither route
currently supports a milestone forecast. No closed family, prior frozen
source or the completed no-refit response-row exception is reopened.

## Exact fixed change and caveats

    cs=(ca+ca.T)*0.5              # stored FP32 symmetric part
    middle=_mm(cs,wa)
    result=_resident_gram512(wa.T,middle)

The real target is W_A.T*Sym(ca)*W_A, not W_A.T*ca*W_A for a general
asymmetric ca. The helper mirrors ONLY the off-diagonal512 block; its two
diagonal512 products retain numerical asymmetry. No full-bitwise-symmetry,
PSD, covariance-domain repair or parent-byte-equivalence claim is made.
Bpre/active_metric MUST be the same new active result. The cold term,
all casts, all16 P heads, CP/star/radial state, supports and thresholds stay
unchanged. No third-plane symmetry or star/CP product omission is involved.

## Synthetic prerequisite: fixed, no network or target data

One old/new pair for each active width1024,992,960,896,832,768, in that
order. Deterministic seed938612+a. FP32 wa~N(0,2/1024); dense ca from an
eight-column low-rank PSD term plus a declared diagonal and tiny skew
perturbation. The896 fixture has a signed diagonal to exercise indefinite
inputs without any PSD repair. These are API/domain fixtures, not choices
among candidate parameters. Exact formulas live in the fixed checker.

Compare BOTH functions to an FP64 product of the stored FP32 operands.
For the new function the reference symmetrizes in FP64; record its separate
FP32-symmetrization error and reference-to-original-operand discrepancy.
Both old/new must be finite FP32(1024,1024), have relative Frobenius error
<=2e-5 and max absolute error<=5e-5 to their appropriate references.
Changed/old relative Frobenius discrepancy must also be<=2e-5 (denominator
new FP64 reference norm floored at1e-30). Strictly lower complete FUNCTION
FLOPs on every fixture, including symmetrization, assembly and allocations.
The common arena is separately billed once, not silently free. This is
not a full estimator bill or a scored experiment. Report wall and RSS;
no kernel/depth/block-size/source/gate retry after failure.

One nice10 BLAS1 OFFPY process, no AP/threadpoolctl, no MPS/concurrent
numerics. Fixed120s total and2GiB peakRSS. Source/protocol SHA binding,
exclusive outputs, no bytecode, finite JSON and preserve all failures.

## Conditional next stage, before target access

Only after PASS, create a source-inverted mechanical candidate and a
byte-exact no-op parent. Change only _cov_compact's active result call,
plus the one helper. AST/import/payload and inverse-source checks.
Fresh matched no-op/candidate processes each evaluate35,100,35 once;
charge initialization once inside BudgetContext, all16 served outputs and
materialization. Parent no-op must reproduce current0c21 all16 bytes and
complete archived bills; repeated35 must be byte-exact with the full
cold-minus-repeat setup difference recorded. Candidate must remain finite,
nonnegative FP32(16,1024), all16 maxabs drift<=1e-5, completeF<=2^41,
each wall<120s and peakRSS<8e9. No runtime guard is weakened.

Seal outputs, complete bills, source/fixture identities before opening
the existing independent1e9 development mean references. Compare each
configuration's OWN raw times max(.1,OWN COMPLETE F/2^41). Advance only
if adjusted improves on both35/100 AND their mean; raw increase is allowed
if paid complete savings outweigh it. No refit, clipping, calibration,
support/threshold choice, alternate checkpoint or post-score retuning.

A passing development screen permits fixed independent14 NC scoring,
both panel groups, all-layer native/repeat checks, NumPy-free persistent
same-backend numerical/complete-bill scoring, wall/residual/RSS and exact
package validation. These remain separate prerequisites before calling
it a qualified component. Keep local/proxy/official results distinct.
Preserve small validated gains; upload only under milestone policy.
