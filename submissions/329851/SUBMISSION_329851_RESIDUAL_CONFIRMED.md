# Submission 329851: confirmed residual-wall-time failure

Matt supplied the terminal card and complete 100-MLP ledger September 4,
2026 at approximately 23:18 PDT. This supersedes the earlier uncertainty
about failure causes; preserve those earlier observations as historical.

All 66 failed MLPs are labeled `RESIDUAL_WALL_TIME_EXHAUSTED`.
34 MLPs completed: 12 with public scores and 22 with sealed scores.
The card reports 12/78 public scored and 66 failed, 0 in flight, 0 pending.
The aggregate official adjusted score is 0.6889726864485142 and raw is
0.6889726907301852; this is unusable as a bank despite `graded` status.

The 12 successfully scored public MLPs average approximately 4.63e-9
adjusted, ranging 3.76e-9 to 5.34e-9, IQR 4.39e-9 to 5.05e-9.
These selected successful runs are not a score for the unscored MLPs.
Mean reported utilization is 20.61%, effective FLOPs approximately 4.53e11.
Completed public MLP wall times range 92.021 to 98.969 seconds.
The numerics/cost are in the expected local class; the confirmed release
blocker is the 0.4-second residual (participant Python) limit, not total
wall time or memory on these listed failures.

Priority: reduce actual Python/dispatch-boundary overhead using ordinary
FlopScope operations and shape-only cached views. Do not change, wrap,
patch, forge, or bypass FlopScope's accounting machinery. Profiling is
diagnostic only; all release tests must run without instrumentation.

New smaller-buffer view-cache candidate, not submitted:
`candidate_fullpair_runtime_d5_t4096_viewcache_v1.py`, SHA256
`882de0bebbe60f7e10f00ac6f362bd71585de0f0d7e507137c2660e09db17989`.
One-thread macOS OpenBLAS rank35: wall 38.900 s, residual 0.156198 s,
max child RSS 5.675 GB, utilization 0.20750319. This is an improvement,
not yet sufficient evidence of official residual headroom. A persistent
five-MLP check is running; no additional submission is authorized by a
mere local pass without stronger margin and exact validation.
