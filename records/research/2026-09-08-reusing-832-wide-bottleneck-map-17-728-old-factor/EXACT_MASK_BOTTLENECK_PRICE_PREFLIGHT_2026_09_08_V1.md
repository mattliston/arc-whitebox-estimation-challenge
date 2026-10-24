# Exact mask bottlenecks: narrow historical opportunity check

September 8, 2026 PDT. Phase 2 only. Root inspected already-saved JSON
shape metadata and performed scalar cost-polynomial arithmetic. No network
or target arrays, new estimator run, factorization, approximation, fit or
submission. No old SVD/rank/age/coalescence closeout is reopened.

## Representation considered

In exact real arithmetic a cohort's linear transport cannot have greater
rank than an earlier active-coordinate bottleneck. If an old group passes
through k active coordinates and later the active count grows to a>k,
one may retain the earlier coefficient matrix Q (k by R) and a later map
U (a by k). Its next factor query is

    pp = (W_active.T U) Q.

This is different from approximating nonzero singular values or discarding
old cohorts. But it pays BOTH the map update and the reconstruction needed
by the current plane reader. New births after the bottleneck are not
automatically in that earlier subspace and must be accounted separately.
Ordinary monotone active-count shrinkage offers no smaller inner width than
the already-compact current carrier by this argument alone.

Reassociation also does not preserve intermediate FP32/Winograd/gate
rounding automatically. A real-arithmetic rank identity is not a stored
factor or all16-output fidelity certificate.

## Cheapest price check, using historical metadata only

Sources:

- difference_leg_cuts_cost_v1.json,
  SHA466840582d6e0f4bed4891bcff1b2174baf93ee7f8f47838b05c9fdab7aeac9b;
- direct_kernel_schedule_prices_v1.json,
  SHA6d0f8d620351b320295302244b6e0f11efdc3402b32235bf2024485c803e66a0.

These are old P0/Q rank35/100 geometry, NOT the current all-P409f
trajectory. Their former-family settings/results are not rerun or varied;
only the preserved active/birth counts and kernel coefficients are read.

Rank100 active counts are nonincreasing, so no strict a>past-min interval
exists for this proposed mechanism. Rank35 active counts are

    1024,1024,1024,1024,992,960,960,896,
    896,832,864,864,832,832,768,768.

The only growth interval is post10/post11 after the post9 bottleneck832.
The old group born through post9 has17,728 signed columns. It can use a
832-wide map while the current active width is864. Later births stay on
their original paid path. For each next pre11/pre12 query, the established
D4 rectangular / eligible D3--5 square cost-polynomial model gives:

| Matrix product | Projected F |
| --- | ---: |
| Original1024x864 by864x17728 |18,942,208,216|
| Reconstruction1024x832 by832x17728 |18,248,156,368|
| Additional map1024x864 by864x832 |829,432,618|
| New sum |19,077,588,986|

The smaller inner dimension saves694,051,848F per transition, but the
paid map product makes the GEMM subtotal135,380,770F MORE expensive.
Across two transitions the difference is270,761,540F, about.05518% of
the archived rank35 first-context490,645,310,609F whole bill.

The pricing follows the already-used8192 column/inner split, charged
concatenation/inner sums, plain fallback when dimensions fail32 divisibility,
and the saved cost coefficients. It is scalar model evaluation, not an
executed FNP kernel or complete current candidate bill.

## Decision and limits

Expected raw change is zero only in exact arithmetic; numerical behavior
is unqualified. Acquisition, retained-Q copies, skipped old gate/materialization
work, map gates, memberships, other changed non-GEMMs, lifetime/RSS and
runtime are NOT included above. Consequently the negative GEMM margin is
not a proved negative complete saving. It does show that the only historical
dimension opportunity is tiny and does not remove a dominant product.
Even deleting the two map products for free would leave only1.388B gross
GEMM savings in this specified group. No complete route to the roughly
16.15% current cost saving needed for4e-9 is substantiated.

No prototype or current capture is selected solely to chase this small
unqualified interval. This does not rule out other exact factor circuits,
nor claim that current all-P masks equal the old metadata. The missing
admission evidence is a genuinely large complete saving with numerical
fidelity; the current proposal does not supply it. The official<1e-9 goal
and raw-accuracy route remain active; frozen artifacts are unchanged.

## Fused-original-group clarification, appended before handoff

The table above is a17,728-column GROUP comparison, not two literal
original full calls. At pre12 the original bank fuses that group with
the1,344 columns born at post10 into a19,072-column inherited push.
Separating the younger group incurs15,921,024F more in the same kernel
price model. The exact changed-GEMM comparison for the two steps is:

| Step | Original affected GEMMs | Proposed affected GEMMs |
| --- | ---: | ---: |
| pre11 |18,942,208,216|19,077,588,986|
| pre12 |20,374,326,064|20,525,627,858|

The second proposed total includes the unchanged-physics but separately
executed young-group product1024x864 by864x1344 at1,448,038,872F.
Corrected two-step GEMM increase is286,682,564F. All earlier non-GEMM,
rounding, old-frame and no-admission qualifications still apply. Do not
reuse270,761,540F as the actual original-versus-proposed fused difference.
