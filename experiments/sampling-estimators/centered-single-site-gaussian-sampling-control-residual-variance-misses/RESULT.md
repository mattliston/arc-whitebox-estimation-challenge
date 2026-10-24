# Centered single-site control: capture far below the required level

The fixed full-network test completed on development ranks35/100 with4096
independent antithetic pairs each. Neither ordinary nor antithetic control
passes the predeclared variance screen. No native candidate, fit, sample
expansion, validation or submission follows.

| Final-layer statistic | Rank35 | Rank100 |
|---|---:|---:|
| Plain path variance | .06738992038 | .07752503438 |
| Antithetic variance | .03045248918 | .03504609891 |
| Ordinary control residual variance | .04165480890 | .04807804257 |
| Antithetic control residual variance | .01869704972 | .02150321506 |
| Reduction versus antithetic variance | 38.6026% | 38.6431% |
| Optimistic classical-IID adjusted variance yardstick, antithetic | 1.087684663e-6 | 1.250930899e-6 |

The antithetic circuit's leading61 matrix-vector products per pair require
variance below3.437954098e-5 for the2e-9 screen, even ignoring all setup and
other arithmetic. Measured residual variance is543.84/625.47 times that
threshold, or1087.68/1250.93 times the1e-9 threshold. Empirical block95%
intervals for the antithetic residual are[.01859827,.01878379] and
[.02135518,.02165496]. This is not a measured native score, certified
nonasymptotic bound or impossibility result for arbitrary QMC, fast batched
arithmetic, nonlinear controls or learned estimators.

The tested stream integrates out a coherent Gaussian reference's auxiliary
noise, propagating every single-site nonlinear residual. Gaussian covariance
uses Hermite orders1..6 with a diagonal marginal-variance tail; within-layer
residual noise is correlated. This extends the diagonal-noise reference in
the prior Astra derivation. The stream and centering idea were already
derived; the new result is this explicit full-network capture measurement.
The known anchor is the reference Gaussian mean, not S31 or the true mean.

Independent convolution quadrature, first-layer identity, small covariance
and residual PSD checks, permutation, and explicit site-sum checks pass.
On the actual eight-input witnesses per network, an independent dense
suffix sum agrees with the streaming residuals within5.78e-15. Independent
scalar fsum variance reconstruction agrees within2.78e-17. Original source
and weight seals remain exact. Declared roundoff-only nonnegative repairs
are recorded, including658/635 entries across all conditional/tail arrays;
they are not unstated positivity fixes or exact floating-point centering.

All four guarded groups completed and retired. Peak sampled RSS658,964,480
bytes, zero added swap. Both sample runs finished in under6 seconds of
numerical work. Saved data are small sufficient-statistic/reference/witness
files. No truth arrays, learned models or validation networks were read.
New Gaussian input samples were generated only for this variance test.

S31 submission332517 remains2.1736465056908115e-9 adjusted in its saved
100/100 official receipt. No fresh poll or new banked gain. Goal active;
2e-9 and approximately1e-9 remain unmet. Submission authorization stands.
