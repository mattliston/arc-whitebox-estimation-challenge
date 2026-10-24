# Prospective unchanged-estimator backend diagnosis

September12,2026, after saved_parity_diagnostic_v1. The candidate is unchanged
4d98fef2. Its local Accelerate independent14 NC score3.793391755502277e-9
is NOT release qualification: fidelity_v2 failed and remains failed.
The fixed all16<=1e-5 and per-net observed-MSE within1% fidelity gates stay.

Two concrete alternatives:

1. Use the exact production OpenBLAS numerical libraries for a native paid
   evaluation of the unchanged estimator. Evidence: same candidate/input,
   deterministic repeats in both backends, between-prediction final MSE
   5.61..7.64e-11, and source-confirmed Accelerate versus OpenBLAS libraries.
   NumPy2.4.6's QR source is identical and internally widens either FP32
   input to double before returning FP32. Expected *mathematical* score
   change is zero; rounding-level score change is unknown, not assumed zero.
   Complete two-call planning cost from actual production35 is403400281707F
   cold plus401455311409F repeat, with the native diagnostic overhead still
   measured independently (historically+15090F/call). No bill is borrowed.
   Unresolved: is the observed difference solely backend arithmetic, or an
   interface/observer/shape semantic issue? Cheapest decisive test is two
   newly paid native35 calls in one process with the exact production
   NumPy/OpenBLAS/flopscope origins, both compared to their already-saved
   production35 outputs under the ORIGINAL all16<=1e-5 gate, plus exact own
   repeat/domain/selector/lifecycle and initialization checks. No truth.

2. Promote shared Gram construction to higher precision. Source tariff
   replay over the saved actual40 scalar geometries estimates at least
   +46.288B mean for Gram action, declared QR64 and explicit casts/scans:
   402.310B ->448.597B, +11.505%, before extra reserve. Passing only QR64
   does not improve QR arithmetic precision, which is already internally
   double, but doubles its billed tariff. Broadly promoting all shared
   construction/transport/read GEMMs is worse (+350.696B before casts).
   No measured raw benefit supports those costs. Unresolved: which Gram
   action/rotation amplifies cross-backend rounding? Cheapest useful test
   would be a target-free fixed replay with subspace/Gram diagnostics, but
   no candidate modification is selected without that causal evidence.

SELECT alternative1. Preserve old receipts and sources. A new OFFPY worker
may explicitly select the existing production OpenBLAS site-packages BEFORE
any numerical import, and must attest actual module/library origins. This
keeps the installed official MLP/materializer and the exact original paid
input/predict/save/reset/release loop. No pip/install/environment mutation,
hidden PYTHONPATH selection, parameter change, clipping or refit.

The new35/35 test replaces an experiment's comparator (old parent-prefix37
versus actual same-candidate production35); it does NOT relax an old test or
qualify fidelity_v2. It is diagnostic only. Any positive result requires a
new same-production-backend native/objective lineage before banking. Actual
raw and complete cost must be measured together, fixed panels/gates retained,
and no Accelerate score may be relabelled as OpenBLAS or official evidence.

Original300s whole/8GB/no added swap/120s retired-progress/<=1s completed
sampling, nice10/sixBLAS1, one numerical tree, exclusive versioned outputs.
Banking still requires all release gates plus validated<4e-9; final goal<1e-9.
The separately selected raw route remains distinct; compute alone is not
claimed sufficient for the ultimate goal.
