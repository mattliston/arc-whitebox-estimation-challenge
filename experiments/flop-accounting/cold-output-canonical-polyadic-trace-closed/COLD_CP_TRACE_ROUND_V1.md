# Bounded cold-output CP read test

September 5, 2026; Phase 2 only. No bank edits or forum posts.

The present card pushes the entire CP bank to ALL output coordinates, then
discards cold coordinates from the carried factors. Cold rows are used only
for the current scalar third-cumulant read. Previous active-pruning rounds
changed carried support; response-row pruning changed the live pair plane.
This test changes NEITHER: retain exactly the .01 active and birth masks,
all factors, all active CP rows/planes, and the existing cold covariance/star
diagonal reads. Approximate only that terminal CP scalar read for cold rows.

For the post-layer CP tensor, form its exact Euclidean partial trace
t=sum_r lambda_r ||p_r||^2 p_r. Project onto Sym(b,C_metric), solving
(tr(C_metric) I+2 C_metric)b=t. Use 24 fixed Richardson iterations, with
step1/(tr C+2||C||F), starting at t/tr C; PSD covariance makes this a
contraction. Cold next-layer diagonal is 3*(W^T b)*diag(W^T C_metric W).
The required covariance diagonal is already present in the parent. A vector
trace and fixed matrix-vector products replace many full-bank output rows.
This is an approximation, not exact tensor compression or truth information.

Small dense trace/projection/linear-push certificate first. One fixed
unlearned screen on ranks35 and100 versus saved parent trajectories and1e9
means; require raw worsening <=3% on BOTH to justify a matched200-row fit.
Report output-row push arithmetic savings as a proxy only. Meter before
any total-cost claim; no lower threshold or extra modes if this test fails.
