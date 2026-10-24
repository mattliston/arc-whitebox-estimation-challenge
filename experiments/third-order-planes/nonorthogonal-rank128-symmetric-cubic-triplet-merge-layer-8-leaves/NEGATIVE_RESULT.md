# Fixed nonorthogonal triplet compression: fidelity gate failed

The prescribed two-sweep rank128 unrestricted CP fit completed. Its final
three factors were explicitly symmetrized into128 mixed cubic triples,
384 stored columns total, without reducing the384-dimensional physical space.
This is a different constructor from the old384 orthogonal-cube projection.

| Consumer | Residual squared energy / original signal |
| --- | ---: |
| 512 reused two-leg Gaussian queries | .5662357525 |
| Complete next repeated-moment plane | .2257116323 |
| Cubic diagonal, secondary | .1282944409 |
| Full symmetrized tensor, secondary | .5920465631 |
| Full unsymmetrized fit, secondary | .6298104239 |

Both primary ratios exceed the fixed.01 limit by a wide margin. The constructor
is closed: no more sweeps, different initialization/rank/ridge, loss-selected
checkpoint, alternative symmetry constraint or native rollout. This negative
does not establish a CP capacity bound or a minimum attainable error.
No physical mean labels, estimator calls, new samples or mean training.

All129 independent small dense checks passed, maximum scaled error3.425e-14.
The actual QR/orthogonality residuals were below1.86e-15 and every regularized
normal equation below4.05e-16. Full signed tensor norm21.1436436215 matches the
old packet diagnostic. Target-merge inner product7.82715642118; symmetrized
merge norm squared7.02869075916. The factors were sealed before verification.

Recorded conventional FP64 GEMM-leading subtotals: construction21,365,784,576F,
qualification1,207,959,552F and diagnostics162,697,052,160F. They omit QR,
solves, elementwise work and buffers and are NOT complete FNP deployment bills.
In particular the fitting cost cannot be treated as free or amortized without
a separately implemented full carrier and complete measured score.

The two guards retired in2.6312s total,peak490,733,568B including self-report,
zero added swap. Session13670 terminal. No candidate, panel, client, archive
or upload. Best official remains Lazy#331599 adjusted3.8164492657795035e-9;
the approximately1e-9 goal remains active and qualified uploads preauthorized.

The algorithmic reference is Kolda, *Numerical Optimization for Symmetric
Tensor Decomposition*, sections3.1/4.7. It motivates unrestricted factor
optimization; it supplies no recovery guarantee for this saved tensor.
https://arxiv.org/pdf/1410.4536
