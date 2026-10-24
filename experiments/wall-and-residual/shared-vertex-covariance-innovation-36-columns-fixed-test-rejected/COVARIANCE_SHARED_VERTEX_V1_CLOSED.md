# Shared-vertex covariance innovation: fixed36-column round closed

September5,2026,15:57PDT. Phase2. No new raw score, candidate or bank.

The three shared-intermediate kernels R diag(h) R (h=1,h2,h3), two
Hadamard powers0/1 and six symmetric(g,phi,alpha*phi) endpoints formed
36 coefficients. All were fit by a single exact normalized Gram solve
with ridge.001, shared across layers8..14, train35/100/150/200/250/300.
Held-out400/500 were opened only AFTER fitting/freezing the coefficients.
Inputs came from the existing honest coherent8M oracle-state source rows;
no new MC or covariance labels were generated.

Conservative deployment-work projection adds9.88%/9.90% FLOPs on35/100,
passing the predeclared10% cost gate. Feature f32-vs-f64 errors on the
first target-free cell35/L8:4.7256e-7,4.6474e-7,4.6492e-7, all below1e-5.
Not native metering or release timing.

Held-out source-information ratios to the zero-correction baseline:

| Network | Covariance MSE ratio | Next-mean-response MSE ratio |
|---|---:|---:|
|400|.995374578823|.998364873060|
|500|.991771573629|1.002680798110|

FAILS the required .75 on BOTH quantities for EACH net. Train gains are
also small (~.4-1.1% covariance), so no epoch optimization bottleneck is
claimed. Normalized Gram min/max eigenvalues .0002623/13.5119; regularized
condition10705.2. No basis/ridge/layer/rank expansion, larger fit, port,
panel or submission is warranted in this fixed dictionary.

Wall11.8248s, peakRSS1,900,544,000bytes, one nice10 three-thread CPU job.
No worker remains. Receipts: covariance_shared_vertex_verdict_v1.json,
covariance_shared_vertex_progress_v1/ (all train/valid Grams, feature
precision, pre-validation frozen fit) and diagnostic model npz. Coherent
sampling is NOT an independent noise-replica correction; no universal
ceiling on joint covariance learning or final-layer MSE is asserted.
