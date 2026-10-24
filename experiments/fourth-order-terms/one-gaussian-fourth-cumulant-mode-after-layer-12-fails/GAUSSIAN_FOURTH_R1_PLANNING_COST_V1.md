# Independent fourth mode: conservative planning inventory, not a meter

September 6, 2026 PDT. Source/shape arithmetic only, independently checked
by root from the efficiency agent's explicit inventory. No estimator,
array, training, or scoring job was run to obtain these numbers.

## Which parent and which hypothetical intervention

This inventory refers to the OFFICIAL D4 bank with prefix P and
SCALAR-FINAL Q, candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py
SHA256 3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757.
It does NOT price the retained response component's full-final P path.

The hypothetical new mode is born post12 only. It changes fourth carry,
not post12 covariance/mean, so original pre13 covariance, gates, support
and birth13 count are unchanged. Post13 feedback can change gates and
supports at14/15. Earlier unchanged-geometry operations stay charged in
the measured parent; affected future products are conservatively bounded
below without subtracting the parent's already-included copies.

## Reproducible gross affected-parent products

Use n=1024 and the existing rectangular D4 polynomial

    M(m,k,r)=a*m*k*r+b*(m*k+k*r)+c*m*r,
    a=1.17236328125, b=19.55078125, c=10.171875.

Width of the14 existing cohorts through post13 is bounded by28n. Current
8192 tiling is charged, including output additions/concatenation:

| Affected product | Gross calculation | FLOPs |
| --- | --- | ---: |
| Pre14 CP plane | 28*a*n^3+(56*b+4*c+3)*n^2 | 36,440,670,208 |
| Pre15 inherited CP push | 28*a*n^3+(32*b+28*c+28)*n^2 | 36,230,856,704 |
| Other mask-dependent products | 10*n^3 | 10,737,418,240 |
| Total | | 83,408,945,152 |

The last row of work comprises star14 readout<=4*n^3, final covariance
diagonal<=2*n^3, final star diagonal<=2*n^3, and the newest shared CP
push<=2*n^3. Star14's two active products plus cold-diagonal product obey
4*q^2*p+2*q^2*(n-p)<=4*n^3. Pre14 covariance/inherited-push and the star's
first product have unchanged geometry and are excluded from this gross
replacement allowance, not excluded from the complete parent bill.

## New mode and odd descendants: safe unoptimized primitive inventory

Classical FP32 V congruences at13/14 and final diagonal read cost10*n^3.
For a straightforward separate signed CP bank (two columns per coordinate
born13 and another two born14), explicit unshared pushes cost12*n^3.
Actual FP64 CP-plane reads for pre14, post13 and post14 cost respectively
8*n^3,8*n^3,16*n^3. This conservatively owns actual stored-atom plane
rebasing, rather than assuming rounded atoms have an exactly equal cheap
analytic plane. Subtotal54*n^3=57,982,058,496 metered primitive FLOPs.
The smaller18*n^3 transport-sharing observation is therefore NOT quoted
as a complete added bill for this unoptimized implementation.

Adding the two known product inventories gives141,391,003,648 FLOPs.
A provisional10,000*n^2 allowance for source/power/elementwise/casts/copies
adds10,485,760,000, totaling151,876,763,648 (~0.152T). Rounding to0.16T
is a PLANNING envelope, not a verified bound: that allowance is not yet
an enumerated implementation, and balanced atom units, exact rounding,
hook/field ownership and lifetimes still require qualification. Adding it
to roughly0.49T suggests around0.65T total only provisionally.

The response full-final path would require its final plane, full
covariance/star reads, and full map/head costs separately. No0.16T complete
bound applies to it. No extra-memory allowance is a measured RSS guarantee;
the retained component already has narrow native RSS margins. All future
costs must be metered in the actual complete configuration and qualified
under the existing memory/residual/wall gates.

## Decision

The cheaper target-free same-source information test is preregistered in
GAUSSIAN_FOURTH_R1_SOURCE_TEST_PROTOCOL_V1.md. Its separate complete native
diagnostic bill will be recorded. No full carrier, refit, raw/adjusted gain,
combination with response rows, or submission follows from this inventory.
