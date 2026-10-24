# Closed: terminal covariance-supervision prerequisite

The fixed covariance-space projection failed both advancement gates.
It captures substantial covariance Frobenius error but little final-variance
response error and does not improve the native final predictions.

| Reference | Projected MSE / parent | Full covariance control / parent |
| --- | ---: | ---: |
| 35, stream A | 1.0039280884164066 | 0.9507067444350218 |
| 35, stream B | 1.0017091930376703 | 0.9520061007263826 |
| 100, stream A | 1.0116283003206332 | 0.9009814335286453 |

These ordinary1B reference scores were computed after all nine native replay
outputs were fixed and sealed. Parent raw MSEs reproduce exactly. The required
projected ratio was <=.80 for every stream. Rank35 cross-stream final residual
ratio is 1.0062472828123523, failing <=.75; the descriptive full-control ratio
is .9284628947343043. This does not establish a population bound or rule out
coherent earlier joint mean/covariance changes.

The target-free active H14-plus-constant basis has rank65 on both networks.
Projected covariance residual energies are .551057/.557276 on35 and .506699
on100 (about44–49% removed), while Gaussian variance-response residual ratios
are .898546/.898493 and .979425 (about2–10% removed). Thus merely having many
covariance labels would not establish useful task information in this fixed
projection. Projection coefficients were privileged per network and chosen
only from covariance errors, not optimized for final means or shared/fitted.

All12 small projector fixtures pass. Actual basis orthogonality/projection
orthogonality pass; affine covariance provenance error is at most4.12409e-5,
under1e-4. Native zero replays pass their original tight output tolerance;
projected35-A repeats exactly in outputs and bill. All prevariances are
positive without strength scaling or repair. Inputs reuse exact accepted-Lazy
TCI captures and original/independent8M references, not new full trajectories.

The old oracle-trajectory P0 decoder, current final-mean bilinear fit and
layer7 free transport remain closed in their original scopes. This result
is a distinct finite covariance-supervision prerequisite on actual cumulative
error, not a new representation family or a successful rescue of those fits.
No covariance fit, rank/cutoff/support/scale variant, wider panel, package or
upload follows. The full reference control cannot advance for the failed
projected arm. No deployable supplier or complete adjusted bill is claimed.

Two guards completed and retired in 3.169551s;
peak observed RSS 636,895,232 bytes, zero added swap. No full estimator calls,
new samples, training or GPU work. Nine native final-block replays plus setup
cost 11,675,304,892 metered FLOPs; covariance reference/projection work is research and
is not represented as a production bill. Disk free 457.88GiB.
Official Lazy#331599 remains adjusted3.8164492657795035e-9. Goal active;
qualified future Phase2 submissions remain preauthorized.
