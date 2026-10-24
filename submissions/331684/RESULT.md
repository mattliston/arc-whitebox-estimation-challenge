# Packed frozen covariance model: qualified and submitted

Submission331684 accepted. Grading is pending; official bank331672 remains
adjusted3.6316074532884024e-9 and the active approximately1e-9 goal is unmet.

This preserves the independent frozen covariance model, its natural amplitude
and layer window, and the original full-lower final readout. Only GNN and
legacy FINAL parameter representation/initializers change. It does not add
the separately fitted readout coefficients. Exact inverse source restoration,
all FP32 parameter bytes, all16 output bytes and complete cold/warm FLOP bills
match the uncompressed covariance source. Native36/38/36+cold38 and ordinary
NumPy-free client cold36/38/37/39/42 plus36/38/36 qualified.

Unchanged-model evidence: panel14 adjusted ratio0.982838708008,
12/14 wins; confirmation8 ratio0.982181085851,
6/8 wins, worst1.025721997737; mean adjusted
3.62422917166e-09 versus3.68998061953e-09.
These are reused public research networks, not pristine holdouts or official
results. No new targets, training or tuning in this representation-only arm.

Matched runtime baseline is the exact official331672 source. Mean cold residual
ratio0.683406108864, maximum packed residual
0.219410453000s; warm sequence[0.2123417800000027, 0.16302641400000084, 0.14890704900000173].
Pooled total-wall ratio1.018301950425; every declared gate passed.
Local timing does not establish official resource completion until grading.
No timing/accounting changes or prediction work outside its ordinary budget.
These conservative cold measurements include module loading; the installed SDK
loads the module before timing predict. Consequently the cold residual ratio
is not a claimed platform residual reduction. See the readout runtime study's
SDK_TIMING_SCOPE.md and prediction-only diagnostic for attribution.

Archive352348B SHA2569fa02622fe0e1fb387594be0ad964c55117d12498c0e5a013557b4a851e1d149;
candidate SHA256facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0. Contract/package pass with
exactly estimator.py and manifest.json. Uploaded once under standing all-uploads
authorization; do not duplicate it. 17 full calls;
19 guards retired, peak6922141696B,
zero added swap. No samples, new datasets or GPU work.
