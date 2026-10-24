# Does the missing third tensor, rather than the whole tensor, have a small sector?

Phase 2, September 5, 2026. New diagnostic, not a deployable estimator.

The previous one-high q4 whole-third replacement failed at rank35/L3:
discarding its all-low sector changed the next scalar third cumulant by
26.8%. That result stays closed. This test neither replaces that tensor
nor changes its rank. It asks whether the ERROR left by the existing full
carrier is concentrated in a one-high sector at a fixed late source cell.
The current oracle-pair attribution identifies off-diagonal third moments
as the main fresh covariance-error channel, but does not identify a cheap
representation of that missing information.

Fixed single cell: rank35, post layer12 -> pre layer13. Fixed q4 geometry
from the current source-v2 pre-target host covariance. Keep the complete
parent CP/star/trace tensor. Geometry and model slices must be frozen before
new truth is acquired. Let U have orthonormal columns and Q=I-U U^T. The
oracle correction is (I-Q tensor Q tensor Q)(T_true-T_parent), with its
coordinate-repeated entries removed exactly before the next W push. This
tests new distinct information, not replacement of known repeated planes.

First certify the inclusion-exclusion projection and the coordinate-star
subtraction against small dense symmetric tensors. Only then acquire two
independent8M streams at this ONE cell, with centered MPS products and
float64 batch accumulation. One MPS worker, no threadpoolctl. For each
stream collect q4 full third slices and fixed32 next-K21 rows (indices
0,32,...,992). No full dense third tensor, fit or full-panel work.

Use the two streams symmetrically to estimate reproducible error and
correction energies through cross-products. Report ordinary cross-stream
natural-strength error ratios as well as noise-separated signal ratios;
do not claim a noise-free confidence bound from only two streams. The
consumer weight is the fixed leading next covariance response computed
from already-stored coherent mean/covariance, not new targets.

Opportunity gate: signal-weighted natural correction must remove at least
30% of the baseline error, correction/error cosine at least .6, and both
cross-stream error ratios below1. Coordinate-plane correction is excluded
from the principal gate and reported separately if diagnostically useful.
Stop after this cell on failure: no rank100, q8/q16, layer, seed or damping
sweep. Success would only authorize a separate source/cost design, not a
learner, candidate or submission; an oracle residual is unavailable at
deployment. No reopening of the closed whole-third compression family.
