# Official Phase 2 launch update

Recorded 2026-08-23 from the official AIcrowd Phase 2 launch announcement:
<https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197>.

- Phase 2 submissions are open after acceptance of the updated rules.
- Submissions close 2026-10-17 at 23:59 UTC.
- The architecture is width 1024 and depth 16.
- The per-MLP budget is `2^41` metered FLOPs.
- The per-MLP wall limit is 120 seconds, the solution-process memory limit is
  8 GB, and the residual wall limit is 0.4 seconds.
- Submission-time numerical computation must use FlopScope; the Python
  standard library is limited to control flow and bookkeeping. NumPy, SciPy,
  BLAS/LAPACK, compiled kernels, foreign-function interfaces, threads,
  subprocesses, multiprocessing, asynchronous work, and computation while a
  FlopScope operation is in flight are prohibited.
- Genuine data files and precomputed artifacts remain permitted.
- The required public dataset revision is `v2-phase2`, with
  `flopscope>=0.12.0` and `whestbench>=0.16.0`.
- The announcement states a Phase 2 team limit of 10 submissions per day; in
  the first week, up to 10 failed submissions per day do not count against the
  normal cap.

The user confirmed that the updated rules have been reaccepted. Submission
authority exists for a future locally validated candidate, but the active
research directive is to reach `1e-8`-class raw MSE with the pure closure
before spending attempts on a hybrid.
