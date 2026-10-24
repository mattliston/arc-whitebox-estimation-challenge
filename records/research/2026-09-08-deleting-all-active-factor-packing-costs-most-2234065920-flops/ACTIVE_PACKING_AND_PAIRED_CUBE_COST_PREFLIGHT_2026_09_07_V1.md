# Active-packed factors and paired scalar cubes: too small or more work

Phase2 only, September7,2026PDT after23:00. Source and existing shape
metadata only. No arrays, model execution, meter, precision experiment,
prototype, fit, score or submission. The efficiency agent supplied the
inventory; root read the effective current predictor, shared push, scalar
cube helper and installed FNP indexing/copy/concatenation source.

Reference: qualified resident_gram512_v1/estimator.py,
SHA256 0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc,
mean complete466705080466.9286F, local independent14 adjusted
4.7768418485455745e-9. No new configuration or score is measured here.

## Exact-layout opportunity and complete affected inventory

Full preactivation pp remains necessary for all output cubics and current
plane/head/radial consumers. But the post-state inherited factors could be
stored as g32[live_indices,None]*right, reusing right=pp[live_indices]
already gathered for the response plane. The newborn base can use the same
ordered active rows. Preserve absolute selected indices, scalar t/s masks
and original operation order on live entries. The next push consumes packed
rows directly. All dense products and live readers remain.

For one post-layer, n=1024, alive count a, newborn-pair count s and incoming
physical columns R, the generously included OLD layout inventory is:

| Work | Charged F |
|---|---:|
| Two FP64 birth copies and diagonal writes | 4n^2+4n |
| Newborn base gather, mask, cast | 12ns |
| Newborn scalar metadata | 32s |
| Inherited gating, physical tails, concatenation, alive mask | 3nR+24ns+3n |
| Next inherited/base advanced gathers | 4aR+4as |

This is 4n^2+3nR+36ns+4aR+4as+7n+32s per post-layer. Installed FNP.12
_ndarray.py at930..941 charges integer-array gather at4 times output
elements, dtype-weighted. FP64 copies/casts cost2 per element; basic slices
are not advanced gathers. Root verified the actual installed paths under
data_gate/.venv/lib/python3.11/site-packages/flopscope, not a guessed numpy
subdirectory.

With a,s<=n and R<=2ln for l0..14, deleting ALL of this inventory gives

    2130n^2+585n = 2234065920F = 0.478689% of the reference mean bill.

Root checked the sum: each layer is bounded by(44+14l)n^2+39n.
This deliberately generous ceiling even includes already-identified dead
newborn tails and assigns zero cost to the replacement. It is not entirely
new opportunity or a measured score improvement.

A concrete packed replacement still has approximately
aR+10as+4a+2n+32s work per post-layer. Applying the agent's ledger to the
historical b2bb rank35 geometry gives old1785042944F, replacement276890496F,
conditional saving1508152448F (0.32315% of current reference mean). This
combines OLD shape metadata with a planning denominator, not current0c21
geometry, a native complete bill or a same-configuration adjusted score.
Post14 stored-factor/base reduction is about27MiB on that geometry; full pp
and large scratch remain. No peak RSS or wall-time reduction is certified.

## Paired cube algebra does not provide a FLOP saving

The exact real identity is

    lambda*(p^3-q^3)=lambda*(p-q)*(p^2+p*q+q^2).

In the same FP64 accumulation/precision convention, including casts,
weighting and reductions, independent cubes cost approximately20F per
row/pair and this literal factorization22F. Favorable free cohort slicing
and halved lambda casting still add217039680F on the historical geometry
before any extra assembly. A precision downgrade is not part of this audit
and would need its own authority/qualification; no such test was performed.
An exact algebraic identity alone does not prove numerical improvement.

## Decision

Neither construction is a milestone-scale cost route. Recommend no
prototype in this round: the layout's impossible free-replacement ceiling
is under0.479%, and the stated cube rewrite increases arithmetic. Preserve
the ownership/cost detail for a future larger compatible construction,
without counting it as a qualified component or reviving a closed precision,
factor-compression, active-congruence or FMM arm.
