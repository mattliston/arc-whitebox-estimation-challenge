# Covariance-innovation independent replica: measurement audit complete

Phase 2 only. Recorded September 5, 2026, 16:57 PDT from completed files,
not the dates in the old handoff. No new estimator, fit, score or submission.

The fixed protocol in `COVARIANCE_INNOVATION_REPLICA_V1.md` is complete.
Rank 35, two independent 8,000,000-sample streams, coherent empirical
previous-layer states, L8 through L14. The original stream's innovations
reproduce the frozen source-v2 targets byte-exactly. The fresh stream used
PCG64DXSM seed xor 0x4153545241434F56 and the unchanged MPS numerical
generator, taking 177.682 s. Analysis took 13.068 s and peaked at 5.819 GB.

Before the current layer's oracle mean feedback, pooled covariance entry
MSE is 2.98746431784e-9; shared-signal point estimate 2.80559400645e-9;
per-stream noise point estimate 1.81870311386e-10, or **6.08778%**. The fixed
target-free next-Gaussian-mean response has observed MSE 6.70899639986e-10
and noise fraction **9.23088%**. Stream cosine is .93913 / .90769 respectively.
Diagonal noise fraction is 9.000%; off-diagonal is 6.08251%.

After consistent oracle mean feedback (equivalently, raw-second-moment
error), covariance observed MSE is 5.50935585443e-9, noise fraction 9.43570%;
next-mean observed MSE 7.30540590595e-10, noise fraction 9.27621%.
These are different channels and must not be interchanged.

The current-parent mean-direction one-high sector contains about 46.1%
of before-feedback covariance energy but only 4.16% of the next-mean
response energy. After feedback those fractions are about 70.8% / 11.0%.
Frobenius covariance geometry is not interchangeable with downstream value.

Verdict: sample noise is too small to explain the near-zero held-out gains
of the closed covariance predictors on this net/protocol. More samples
alone are not the missing-information fix. This does not reopen any of
those predictors. The predeclared rank-100 gate (at least 20% pooled noise
in either before-feedback channel) is false. **No rank 100 run or seed /
sample-size sweep follows.**

Qualifications: two streams give point estimates, not confidence-certified
ceilings. Shared MPS/float32 numerical bias is unmeasured; nonlinear
empirical-state feedback can add O(1/N) bias. This is not a final-layer
score or a global impossibility claim.

Receipts: `covariance_innovation_replica_r35_v1.json`,
`covariance_replica_truth_v1/rank_0035_8M_v1.npz` (SHA256
e6a7787d974c6b56f39a4444bd54c793ff3efa6e62acb55054774ddb2a2e3358),
`covariance_replica_truth_v1/innovations_rank_0035_v1.npz`.
