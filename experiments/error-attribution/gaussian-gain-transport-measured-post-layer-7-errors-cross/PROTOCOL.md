# Lazy covariance-error transport: fixed privileged screen

Phase 2 only. Prospective on 2026-09-20, before this study decodes targets.
Parent is the exact submitted Lazy source b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c.
Official #331599 stays unchanged. Qualified future uploads are preauthorized.

Question: does a measured mean/covariance error at post layer 7 contain
useful information about final error under the already specified Gaussian
gain tangent, on the actual Lazy trajectory? The older matrix observer
learned sources from final mean labels; older nonlinear covariance decoders
used oracle previous mean/covariance. Neither is this privileged recurrence
measurement. This is not an unrun claim about either completed experiment,
a new learned supplier, or a full Lazy derivative.

Passively acquire the original native 35/100/35 sequence. At post layers
7..14 retain internal corrected mean and masked covariance; at pre 8..14
retain covariance and scalar Gaussian response data; at final 15 retain
internal mean, scalar variance and response data. No numerical operation
is added to the parent. Require all 16 returned means and complete original
cold/warm/repeat FLOP bills byte exact, repeated captured arrays exact, and
W5 numeric caches retired. Seal captures before decoding target arrays.
Cap captured files at 500,000,000 bytes. Keep the original archive intact.

Use canonical weights from the observer_fit store. Their bytes must agree
with the old covariance source's row weights. Use the existing coherent
8M post means and full post covariances for ranks 35 and 100, plus the
independent rank35 8M stream (seed xor 0x4153545241434f56). The two original
cov/cov2 names are not independent streams. Pin source code, sidecar and
all truth files before decoding. Require P=8000000 and the stored post7
to pre8 affine mean/covariance consistency at relative error <=1e-4.
This qualification is about matched data, not MC uncertainty.

Let e_l=true_mean_l - internal_parent_mean_l and
E_l=sym(true_cov_l - masked_parent_cov_l). Do not center covariance around
the parent mean: each source is already centered around its own mean.
Report parent covariance antisymmetry separately. At each subsequent layer
use the following fixed tangent, with recorded parent Cpre, sigma, alpha,
gate g, pdf p and Gaussian mean mg:

    u = W.T @ e; V = W.T @ E @ W; z = diag(V)
    e_new = g*u + p*z/(2*sigma)
    dg = p*u/sigma - alpha*p*z/(2*sigma**2)
    E_new = (g g.T)*V + Cpre*(dg g.T + g dg.T)
    diag(E_new) = (g - mg*p/sigma)*z + 2*mg*(1-g)*u

Symmetrize E_new. No cold mask is imposed on this auxiliary error matrix.
It includes true correlations omitted by the primary mask. This is exactly
the old Gaussian gain approximation, not the full bivariate Gaussian map,
cumulant/selector/GNN derivative, or a claim of physical state consistency.
At final15 only the mean equation is needed. Apply its formal positive
branch using the recorded internal final mean, and add no learned source.
Test the tangent against central differences of its stated Gaussian-gain
map on fixed small SPD fixtures, relative <=1e-6, plus exact linearity.

Exactly two free-propagation arms start at post7: (a) measured e7 and E7;
(b) measured e7 and zero E7. Arm (b) still generates subsequent covariance
error through the same equations. There is no later truth reset in either
arm. Also report descriptive one-step transport from true e/E at every
preceding layer8..14, to distinguish accumulated from local error; these
one-step values are not free forecasts or scores.

For rank35 independently form both arms and residuals with stream A and B.
Report cross-products mean(residual_A*residual_B) and observed/noise identity
(mean(A^2)+mean(B^2))/2 = mean(A*B)+mean((A-B)^2)/2 for mean/covariance fields.
These are point estimates under independent sampling streams, not confidence
bounds; shared numerical bias and empirical covariance O(1/N) bias remain.
For both networks also report ordinary final MSE against the existing 1B
mean reference, with all forecasts fixed before reading that reference.
Those references are reused development data, not independent validation.

Advance only if rank35 cross-stream final residual has positive parent and
mean-only denominators, full/parent <=.75 and full/mean-only <=.75, AND
rank100 ordinary 1B residual satisfies the same two ratios. A pass only
justifies designing a separately qualified supplier; it is not a candidate,
cost improvement, adjusted score, permission requirement, or submission.
A failure closes this fixed layer7/recurrence screen. No cutoff, response,
scale, rank, feature or fitting rescue follows in this experiment. No
training, new samples, native state writes, or upload during this screen.

One numerical process group at a time, nice10, all six BLAS thread controls1,
8GB group RSS, zero added swap required. Existing run_guarded.py enforces
8GB with 256MiB emergency swap termination. Unit30s, capture180s, analysis90s.
The approximately1e-9 goal remains active regardless of this screen's result.
