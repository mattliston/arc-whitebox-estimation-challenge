# Public third-plane endpoint span — fixed objective CLOSED

2026-09-06, finish04:46:11PDT; no-refit bound finish04:49:08PDT.
Phase2 only. Actual bank and official score unchanged; no new submission.

The newly discovered public100M joint-moment dataset is numerically
qualified on the preselected first training mini network57, gregory-wilson.
Its pinned337MB file passed SHA256 and identity checks. Source seed/sample
count match the contributor card. First Gaussian-layer standardized RMS
errors are1.00116 for covariance and1.00025 for K21. Pre13 affine mean
discrepancy8.96e-11 RMS, covariance2.66e-6 relative. Stored-f32 K21 rounding
bound is .00028057 of on-policy missing-plane RMS. This is not a bound
on finite-sample noise or shared GPU bias. The 100M post-mean array was
decoded for centering/identity; official1e9 gt_mean was NOT decoded, and
no final score was calculated. No held-out moments were opened.

Before labels, actual bank prefix and current64-state features froze.
Independent pre14-response replay reproduced all15 input/prefix arrays
byte-for-byte. Weighted adjoint and small dense ridge solution certified;
wide third-to-covariance kernel agrees with exact substitution to2.17e-19.
F=[1,H64] has rank65, span reconstruction1.48e-15. Three rho powers,
unrestricted65x65 coefficients, weighted K21 loss, ridge.001 were fixed.

The300-iteration solve did NOT pass tolerance: normal residual2.158887e-5
versus1e-7. Its in-sample isolated-response error ratios were:

- weighted off-diagonal K21 plane: .5523343359;
- covariance response: .4689963194;
- following-mean response: .8595619870.

No extra iterations followed. A separately certified posterior bound uses
A >= .001 I and the exact quadratic-response Gram row-sum norm bound.
With1.001 safety inflation, the UNIQUE EXACT optimum of this same ridge
objective still has next-mean error ratio >=.529467652, above the unchanged
.50 gate. Thus that value gate cannot be rescued just by finishing this
solve. The initial bound script failed only on JSON NumPy-bool serialization;
v2 preserved it and repaired serialization, with zero refitting iterations.

This closes the fixed endpoint dictionary/objective test, not all third
predictors or an unregularized universal representation. Single-network
in-sample oracle coefficients are not cross-network learned predictions,
total covariance errors, final raw MSE or adjusted scores. No remaining
seven-file download,6/2 fit, parameter/objective/feature sweep, state write,
port, panel or upload follows under this design.

Preserved receipts: public_p2_moments_metadata_v1.json (fixed potential
train57,68,45,59,3,12; held-out14,71; all mini seeds disjoint from full1000
and virgin6), public_p2_first_moment_qualification_v1.json/.npz,
public_p2_third_endpoint_span_v1.json/.npz, third_endpoint_solve_bound_v2.json.
The qualified dataset remains useful for a genuinely distinct future model.

Public source and attribution:
https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026
commit1bb00b739619154ca255db6443af0b6a9dd960ce, contributor MIT card;
original WhestBench v2-phase2 CC-BY-4.0. Source file SHA256
f2c79442c4adbf4e12a7db98f6a72a74c66d9c50872f99f8ff63599834d72670.
