# Saved covariance fit: backend mismatch ruled out as sole cause

The four stored MPS training-endpoint and native submission predictions differ
by only 0.217%–0.338% of native error RMS. Using the prior sealed native/P-only
MSE scalars and the triangle inequality, training-endpoint MSE is at least
2.0444, 2.4082, 1.2732 and 1.4116 times P-only for networks37/39/42/48.
Thus this exact saved endpoint failed on the training backend too. Backend
mismatch alone cannot explain the old joint P/Q result.

The matched batches also show all14 post-initial online joint losses exceeding
P-only, after a 24.03x jump on batch2. These are changing-parameter batch losses.
No optimizer bug, native-gradient accuracy, convergence or capacity bound is
established. The old fixed fit remains closed; no learning-rate/checkpoint rescue.

All same-payload, source and canonical-weight links verified. Exactly8 old
prediction arrays decoded; no truth arrays, new model calls, fits or samples.
The single worker retired after1.050s, with38.9MB sampled group RSS and zero
added swap. No candidate/archive/upload. Best official remains Lazy#331599
adjusted3.8164492657795035e-9; approximately1e-9 goal remains active.
