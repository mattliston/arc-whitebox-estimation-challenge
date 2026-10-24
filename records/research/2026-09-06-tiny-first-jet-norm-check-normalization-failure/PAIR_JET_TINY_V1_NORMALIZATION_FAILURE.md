# Tiny first-jet v1: norm-check normalization failure

September 6, 2026 PDT. One escalated nice10 / one-thread official-Python
worker terminated exit1, before the static cost calculation. The frozen
source is `certify_pair_jet_coalescence_v1.py`. No network/card state or
target data was loaded by the tiny stage; no generated JSON was written.

Fixtures0-6 and8 passed every gate. All nine tensor, repeated-plane and
next-layer star-restored identity checks pass the unchanged1e-11
relative-to-reference gate; maximum tensor identity error4.0483541151e-13.

Fixture7 deliberately has pair difference norm1e-5 and transverse norm.1.
Its solved cube is a very BAD approximation: ||E||/||T||=2462.335006.
It would unambiguously fail the fixed <=.1 eligibility criterion.
The Frobenius error-norm formula's floating-point discrepancy, when
normalized by the much smaller ||T||^2, is2.1875847410e-9 and fails1e-11.
Relative to ||E||^2 this discrepancy is about3.61e-16, i.e. ordinary
floating-point roundoff. The tensor identity on that fixture still
passes at4.05e-13 relative to T.

Proposed v2 diagnostic change, pending root approval: report both norm
normalizations, retain the original norm-error-to-T gate for every
eligible pair, and check the norm of huge, ineligible errors relative to
that error's own norm. Keep every algebra/plane/star identity gate,
all fixtures, first-jet construction, eligibility and cost schedule
unchanged. This is a correction to a numerical test scale, not permission
to merge unstable pairs, relax the .1 approximation criterion, or run
any wide state/parameter sweep. Preserve this failure and the v1 source.
