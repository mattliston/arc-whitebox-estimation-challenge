# Phase 2 repaired bank: submission 329851

Accepted at 2026-09-05 05:28:19.210 UTC (September 4, 22:28:19 PDT).
Acceptance response: submitted; Successfully enqueued 1 Job; no scores yet.
Challenge 1174, Phase 2 round 1429. Upload was explicitly authorized by Matt.
This is an immutable acceptance snapshot, not a passing official result.

Candidate: `candidate_fullpair_runtime_d5_t8192_v1.py`.
Source SHA256: `93105a85c1b7a1222fc4dea732dd76b3568d607e5e0bead5fffb94c6c72b72d6`.
Archive: `fullpair_runtime_d5_t8192_v1_whest016.tar.gz`.
Archive SHA256: `34d2aeb83996a8f5f44f41ce5e4c2e84548e829016ec296daa726df6b8222c8d`.
Release audit: `verified_local_fullpair_runtime_d5_t8192_v1.json`.

The previous bank 329843 failed every attempted MLP with TIME_EXHAUSTED
(120-second per-MLP limit). Nothing scored or counted against the quota.
The repair changes rectangular multiplication to D5/8192 and caps square
recursion at depth 5. It retains the full factor bank and both learned heads.

Independent 4M-target validation: public 8 adjusted 5.067342855116864e-9;
virgin 6 adjusted 4.110753451886179e-9; all 14 adjusted
4.657375968017999e-9. Combined raw MSE 2.258865033944123e-8 and mean
metered utilization 0.2057857142857143. These are local, not official scores.

Actual NumPy-free client / official-version server tests, one BLAS thread:
rank 35 wall 34.518 s, residual 0.191471 s, peak 6.913 GB;
rank 100 wall 36.510 s, residual 0.191539 s, peak 6.633 GB;
virgin rank 3016500 wall 35.131 s, residual 0.191390 s, peak 6.590 GB.
Repeated native 35/100/35 parity and memory tests also passed.
These stronger local margins do not guarantee official Linux-worker timing.

No second upload should occur while 329851 remains pending. Official best
remains submission 329790 (adjusted 1.436993412541317e-8) until a passing
result supersedes it. The below-4e-9 official goal is not yet achieved.
