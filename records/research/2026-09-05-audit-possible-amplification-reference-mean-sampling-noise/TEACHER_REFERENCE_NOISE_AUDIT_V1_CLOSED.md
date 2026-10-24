# Reference-mean noise audit: closed at the tested scale

2026-09-05, after 12:31 PDT. Phase 2 only. No estimator or training change.

The two unchanged teacher-forced baselines are byte-identical. Under the
explicit iid Gaussian-input N=1e9 reference-mean noise model, eight fixed
second-moment probes estimate layers 10–15 noise/residual MSE fractions of
0.0032950 on rank 35 and 0.0026949 on rank 100 (0.330% and 0.269%).

Final-layer modeled noise MSE is 1.20606e-11 / 1.01182e-11, compared with
frozen-head teacher residual MSE 5.43848e-9 / 5.45859e-9, respectively.
Mean late nonlinear-even response MSE is 2.72237e-13 / 5.87799e-13.
Each two-sided eight-direction audit took approximately 110 seconds.

This is far below the predeclared 10%-on-both closing threshold. It does
not support changed teacher targets, denoising, more noise probes, or a
claim that reference noise explains the failed feature probes. Actual
sampler provenance remains unverified: the stored files have no sample
count or sampler metadata. These are model-based sensitivity numbers,
not measured reference error or a universal noise floor.

Receipts: teacher_reference_noise_r35_v1.json and
teacher_reference_noise_r100_v1.json; protocol in
TEACHER_REFERENCE_NOISE_AUDIT_V1.md. No new official submission.
