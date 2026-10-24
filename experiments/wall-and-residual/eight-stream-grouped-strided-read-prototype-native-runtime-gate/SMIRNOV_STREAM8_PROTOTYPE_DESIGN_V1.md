# Fixed native Smirnov stream8 product prototype

Phase2, September6,2026. The executable plus border-copy envelope passed
the20% gate on both current censuses (20.0598/20.2591%). This licenses
one product prototype, not a full candidate.

Fixed largest rank35 geometry:1024x832 times832x8192. Fixed schedule:
three executable primary3x3x6 Smirnov stages, then one2x2 Winograd stage
with the exact one-column inner fringe. Transform the top40 products
once, process them in groups of8, and combine them once. Reuse the
already-certified streamed7-product Winograd leaf as a component; do
not rerun the old leaf alternatives or claim this leaf itself is new.
Its three input identity copies per side disappear and must be reflected
in the actual metered bill. All non-leaf arithmetic uses the exact new
primary executable DAG. Only partial input blocks are zero-padded and
copied; output border crops must be written/costed explicitly.

Use one reusable1GiB scratch arena with an interval allocator and exact
nonoverlapping lifetimes. Track dedicated zero-padding buffers separately;
total retained work allocation must stay<=1,168,900,096 bytes. Do not infer
this from the final leaf A+B+C footprint. Count all nativeFNP.12 work,
including input/output copies and fringe additions. No numpy inside the
product; NumPy is allowed only for external reference inputs/checking.

First a small shape with the same3-stage/streamed-leaf structure must
match float64 NumPy within relative2e-4 and remain finite. Then fixed
full geometry, seed930066, normalized iid Gaussian A/B, two repetitions
(cold,warm), one and four BLAS threads, compared with the existing bank's
exact same product operation. This two-thread-count pair is fixed, not
an optimizer sweep. Require relativeFrobenius<=2e-4, work cap, and warm
runtime<=3x the same bank product at BOTH thread counts. Failure closes
this fixed circuit/grouping prototype before actual-network input tests.

A pass would STILL need actual structured CP inputs, complete metered
estimator bills,14-net independent panels and persistent total/residual
time plus memory. The tiny projected4e-9 margin is not an official score.
No matrices from gated/private challenge nets, no new MC, forum writes,
commits, submissions, or changed frozen candidates.
