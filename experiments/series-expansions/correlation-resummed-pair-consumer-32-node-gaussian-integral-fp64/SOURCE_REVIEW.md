# Source and formula admission

The current parent assigns nonlinear=_nonlinear_pair_batch: Gaussian Mehler6,
Edgeworth correlation2, local terms and K31 included. Its first Gaussian
pair helper uses order6 too; final prediction already uses scalar moments.
The marginal and post-centering code will be copied exactly.

radial_mehler_r35/r100_v1 and screen_radial_mehler.py really executed old
Gaussian6/edge2 variants. They did not test higher correlation orders. A
bounded Astra source/text search for higher edge_order or exact-correlation
Edgeworth found no matching executed route. This is not a universal novelty
claim. Prior higher-cumulant, fourth-source, L3-squared, Gaussian mixtures and
current native training negatives remain unchanged.

For standardized positive-part powers, Gaussian integration by parts gives
M_pq = a M_(p-1,q) + (p-1) M_(p-2,q) + rho q M_(p-1,q-1),
with boundary additions Bx_q if p=1 and rho By_(p-1) if q=0.
The code uses only p,q<=2. For distributional derivatives beyond the power,
f^(p)=power! delta^(p-power-1). Conditional Y|X=0 has mean b-rho*a and
variance1-rho^2; differentiating its density-weighted moments supplies the
required delta derivatives through order2. The probability integral uses
d Phi2(a,b;rho)/d rho=phi2(a,b;rho), integrated from independence.

Primary mathematical context: Felix Voigtlaender, A general version of
Price's theorem, https://arxiv.org/abs/1710.03576, and the organizers' cumulant
framework https://arxiv.org/html/2605.05179v1. These support distributional
Gaussian differentiation/cumulant methodology, not a competitive-score claim.
The explicit recurrence and implementation are our derivation, independently
checked by direct conditional integration and Hermite sums before deployment.

Literal native rho may have rounding antisymmetry. Both boundary orientations
use the same entry rho_ij; replacing one by a transpose would be wrong for
that input. This was corrected in unsealed drafting, before any numerical run.

No true means, model state arrays, fit, new samples, candidate execution or
external message has occurred at this seal.
