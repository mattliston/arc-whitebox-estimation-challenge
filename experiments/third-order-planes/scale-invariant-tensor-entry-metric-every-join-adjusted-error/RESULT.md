# CP entry norm weighting fails the native endpoint test

The fixed current-S31 candidate worsens pooled adjusted error by0.88223%
on development ranks35/100, with0/2 adjusted wins. The full native bill
increases only0.005125%. This closes the specified endpoint follow-up;
no TRAIN6, refit, metric/exponent/window selection, composition, validation
or upload follows.

| Rank | Raw MSE change | Full bill change | Adjusted change |
| --- | ---: | ---: | ---: |
| 35 | +0.73363% | +0.005121% | +0.73879% |
| 100 | +1.01139% | +0.005130% | +1.01657% |
| Pooled | +0.87725% | +0.005125% | +0.88223% |

The entry weight is lambda^2 ||p||^4 at the existing joins, with all
rank/schedule/native routing/learned parameters unchanged. J^2 is reused
between the norm reduction and the existing row-energy selection; all
new arithmetic is charged. The source changes one operative function.
Candidate SHA52ae1b05925d958d9e26bd9fbae835bc60cd6b09e5b4fa315a9bffe81effde56.

This was explicitly an exploratory endpoint follow-up to checkpoint13,
not a new method or a reclassification of that result. The prior3.84%/3.36%
full-tensor projection improvements still fail their prospective20% gate.
No endpoint test existed then; it now exists and is negative. Better
fidelity in that unweighted tensor norm did not improve these final means.

Native small-metric arithmetic, selected rows and power-of-two CP gauge
checks match NumPy exactly. First7 output rows are byte-exact to S31 on
both networks. Full predictions are finite16x1024, and offline scalar
math.fsum independently reconciles every MSE and adjusted score against
the existing1B references. These are reused development networks, not
fresh validation. Complete native bills are281870906091 and281084831942.

All three guard groups completed and retired. Peak sampled RSS5280776192
bytes; zero added swap. The native evaluation session67669 was reaped
with exit0. Source/protocol/evaluator/parent/control hashes were sealed
before evaluation. Official S31 and its submitted archive are unchanged.

Read-only prior review also confirmed: the exact full terminal Gaussian
fourth-tree readout has an executed0/2 adjusted-win predecessor; the
input-radius/angular closure ran a matched fit and had only about0.8%
development gains. Neither was rerun or presented as a new experiment.
