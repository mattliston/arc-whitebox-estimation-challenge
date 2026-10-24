# Mixed rectangular multiplication: optimistic envelope, not a bank

September 5, 2026, 17:51 PDT. Phase 2 only. No estimator or submission.

All six cyclic/transpose versions of the primary Smirnov 3x3x6 rank40
tensor passed exact rational basis-product certificates. Transposed linear
program operation counts use the linear transposition principle. They do
not constitute implemented kernels, copy counts or runtime measurements.

A bounded dynamic program mixes these six shapes, current packed Winograd,
and the published accurate rational 4x4 rank48 arithmetic counts. Maximum
five recursion stages, minimum leaf dimension eight. It uses the existing
byte-exact bank matmul census for ranks35/100; no new estimator run.

Pad-only v1 offers optimistic whole-card savings18.043%/18.484%. Version2
adds exact classical fringe peeling at every stage, with inner-fringe
accumulation additions included, giving19.7082833%/19.9162497%. At the
unchanged official raw score, the4e-9 threshold needs19.7146895% savings.
Even the optimistic envelope therefore only brushes that threshold.

Requiring necessary full-batch leaf A/B/C storage to fit the current arena
reduces the v2 envelope to17.5137576%/17.5853781%. This remains only a
memory lower bound, not complete live storage. New-algorithm copies,
allocations, remote overhead and full scheduling memory are NOT charged.
No practical saving, runtime gain, raw score or threshold bank is claimed.

Do not build a production kernel on this arithmetic envelope alone. Keep
the certificates as potential complementary savings if an independently
strong raw lever appears. This is not a universal lower bound on matrix
multiplication or a prohibition on genuinely different scheduling methods.

The broad scalar integer-power receipt was also re-read: relative to the
already-selected CP-cube rewrite it saves only13.787M FLOPs on rank35,
while worsening remote wall85.10s to94.27s and residual .36836s to .41537s.
It is not an untested several-percent cost lever.

Receipts: smirnov_cyclic_certificate_v1.json, mixed_fmm_envelope_v1.json,
mixed_fmm_envelope_peel_v2.json, and preserved scalar-power remote records.
Primary sources: https://fmm.univ-lille.fr/3x3x6_raw.mpl.bz2 and
https://arxiv.org/html/2603.18699v3. The latter's alternative-basis47 inner
products have nontrivial change-of-basis cost; no uncharged47-product
implementation is inferred.
