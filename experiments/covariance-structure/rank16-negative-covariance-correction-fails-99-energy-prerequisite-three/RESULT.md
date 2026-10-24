# Current negative covariance modes: closed before accuracy scoring

The fixed rank16 action fails its spectral-energy prerequisite on both
research networks. These are the three failed cells out of fourteen:

|Rank|Post layer|Negative energy retained by16 modes|
|---|---:|---:|
|35|10|0.987867190709|
|35|11|0.984375375637|
|100|13|0.972735234286|

Every cell required at least0.99. No rank, threshold, diagonal scaling,
gain, layer or fit rescue follows. score.py remains unexecuted, and no
8M covariance labels or ordinary1B means were decoded. No native candidate,
complete candidate bill, accuracy score, package or upload was produced.
This is not evidence of a failed accuracy gate, because that stage did not run.

The negative component is small: its full Frobenius norm is
6.44328e-08 to 3.77206e-07 of the actual covariance
norm over these cells. This describes the current saved331684 trajectory;
it does not establish a general PSD or accuracy guarantee. Original
negative counts range32..129; signs very near zero can be sensitive to
numerical precision. The fixed squared-energy gate is independently stable.

The audit uses an eigenvalue-only solver on the opposite matrix triangle.
Spectra agree within 2.55562e-15 relative, and retained-energy fractions
within 3.69774e-11 absolute. Selected factors satisfy the original
eigen-equations and orthogonality. Dense and factored covariance-to-mean
responses agree. Small PSD/zero/indefinite, permutation, orthogonal-change
and rank-selection tests pass; Gaussian derivative finite differences agree
within8.13e-11 relative.

A separately labeled post-result, target-free geometry bound gives
2.66523e-07 to 1.72375e-06 RMS for the immediate Gaussian mean
response of the FULL negative spectral part. It follows from its spectral
norm and the next weight-column norms. This is not a measured true-error
reduction or bound on learned-head/recurrent effects, and it does not
change the failed gate. No additional estimator action was selected.

For context, full negative spectral clipping of a symmetric matrix is a
standard positive-semidefinite projection construction ([Higham,1988](https://eprints.maths.manchester.ac.uk/363/)).
Our proposed C+D preserves the parent's antisymmetric part, changes the
diagonal and keeps at most16 modes; it is not that full projection or the
earlier diagonal-preserving repair. Offline eigendecompositions do not
establish an affordable inference algorithm. The existing symmetry-only,
high-order, covariance-fit, FMM and compression results remain unchanged.

Three guarded groups retired, zero added swap. Peak sampled group RSS
444219392 bytes; self-reported peak
444219392 bytes. Artifacts before closeout
total 2033899 bytes. No new forward
prediction, samples, download, fit or GPU work. Existing captures/weights
were reused by reference without duplicating the datasets.

Official331684 remains adjusted3.5844144212297072e-9,100/100 complete,
zero failures.4e-9 is banked;3e-9,2e-9 and approximately1e-9 remain unmet.
The original goal stays active. This closes only the declared spectral
action and does not prove every covariance correction ineffective.
