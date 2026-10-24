# Saved joint-covariance optimization and backend-gap diagnostic

Phase2 only. Preserve all old fits and failures. No new candidate, model
execution, fit, samples or target-array access. Written before decoding
the saved predictions for this diagnostic.

Source review of current_comp_joint_p_covariance_v1 shows the old fixed
P+Q fit really ran60 backwards/15 updates and failed native CHECK4 at
adjusted1.7725560566 times its matched P-only arm. Its scalar logs show:
the first batch has identical raw mean and P gradient norm; after one
update the joint/P-only raw ratio on the next matched batch is24.0301861.
All14 later online batches are worse. These observations do not identify
the cause or demonstrate that a different fit would work. Source ownership,
nulls and execution integrity passed; no implementation bug is presumed.

Question: can the discrepancy between the old training backend and native
inference, by itself, plausibly explain that terminal failure? Read only
the four previously sealed MPS terminal predictions and their four native
counterparts for37/39/42/48. Authenticate same terminal P/Q payloads,
literal/native build links and canonical weight hashes. Reuse only the
ALREADY published per-network native/P-only raw MSE scalars from its sealed
score receipt; do not open any truth file or compute a new candidate score.

Let e be the saved native RMSE and d the RMS distance between the paired
MPS/native predictions. By the triangle inequality, the MPS RMSE against
that same old reference lies in [max(0,e-d), e+d]. Square both ends and
divide by the saved P-only raw MSE. Report every network. If all four lower
ratios exceed1, then the exact saved training-model endpoint also worsens
all four old references: backend mismatch alone cannot explain the failure.
Otherwise this diagnostic is inconclusive on that explanation. Numerical
FP64 evaluation of this identity is not an interval-arithmetic certificate,
a native-gradient check, an official score or general model-capacity bound.

Also preserve all15 same-batch online raw ratios and both parameter-group
gradient norms/LRs from existing JSON, distinguishing moving parameters
from fixed-endpoint validation. Initial P norm equality is not claimed as
full gradient-byte equality. No causality or convergence result follows.

One guarded nice10/BLAS1 process,30s,8GB RSS,zero added swap. Decode exactly
eight small(16,1024) FP32 prediction arrays. Write only scalar JSON, no new
arrays. No step-size, epoch, covariance-strength, head or checkpoint rescue
of the old fit follows automatically. A future different action requires
its own prospective data, solver, complete-cost and validation definition.

Current official best remains Lazy#331599 adjusted3.8164492657795035e-9;
the approximately1e-9 goal remains active. Qualified submissions remain
preauthorized. This diagnostic cannot qualify an archive or upload.
