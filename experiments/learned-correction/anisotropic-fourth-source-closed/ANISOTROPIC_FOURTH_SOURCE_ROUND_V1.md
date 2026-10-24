# Bounded anisotropic fourth-moment source test

September 5, 2026. Phase 2 only; bank329884 untouched and no forum posts.

The previous diagonal-trace Sym(C,D) fourth model on the older plain host
was stable but only modestly useful. Its off-diagonal innovation fit was
unstable and is CLOSED. Do not repeat that fit. This round asks a distinct
question on the current full-pair host: does retaining diagonal partial
traces WITH the corresponding fourth-to-third source improve raw error?

Carry D=alpha C+E, fit only D's diagonal to available post-K22 partial
traces, and keep E on the existing active support. The consumer uses the
actual K4=vi Dii, K22=(vi Djj+vj Dii+4CijDij)/6 and
K31=(vi Dij+Cij Dii)/2; not the scalar-radial K31 formula. No oracle inputs.

For the quadratic ReLU vertex, the anisotropic extra cubic source is
[Sym(d*diag(C),gEg)+Sym(d*diag(E),gCg)]/12 plus coordinate mixed-column
births. Project ONLY the new vector-matrix source's partial trace into the
existing covariance-trace carrier. This projection is an approximation,
not an exact full-tensor source. For the mixed-column birth, shift its
existing covariance column by E/[6(1+alpha/3)]; this matches the new source
to first order in E, with explicitly measured quadratic remainder. No extra
CP bank factors are introduced. One extra active covariance-like push is
needed; meter only if accuracy earns it.

Certify plane formulas and first-order mixed-column identity on small dense
tensors. One fixed full-strength unlearned screen on ranks35 and100 versus
saved current parent and1e9 means. Require >=15% raw improvement on BOTH,
and a relative birth-column shift below .1, before a matched200-row fit.
No off-diagonal innovation fit, damping search, threshold sweep, or head reuse.
