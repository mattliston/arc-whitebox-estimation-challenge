# Active congruence512: fixed synthetic prerequisite failed

Phase2 only. Completed2026-09-08T05:17:07.922186UTC, September7,22:17:07PDT.
One nice10 BLAS1 OFFPY worker,1.8617113s total, peakRSS604880896B. No
network/target arrays, estimator rollout, head fit, candidate or upload.

The first fixed active1024 fixture completed both function measurements:

| metric | parent | changed |
| --- | ---: | ---: |
| complete function FLOPs |2394771456|2155137024|
| wall seconds |.13345454|.23416662|
| relative Frobenius error to declared FP64 reference |1.99643155e-5|2.06238527e-5|
| max absolute error |4.16556739e-6|3.80393051e-6|

Changed/parent relative discrepancy2.630226021484745e-5. Both changed
reference error and changed/parent discrepancy exceed the fixed2e-5
gate. Strict cost reduction239634432F exactly matches source pricing,
but does not override numerical failure. The remaining five fixtures did
not run. No claim is made about their errors or all-shape cost validation.

Original covariance asymmetry2.54664124e-7 relative; symmetric/original
FP64 reference difference1.25775840e-7; FP32 symmetrization reference
rounding2.94289303e-8. The changed output's numerical asymmetry remains
1.95880668e-5 because only its cross512 block is mirrored. These are
diagnostics, not excuses to change the gate or call the output PSD.

Genuine FNP operands were supplied to both functions. Their separately
recorded ingress charge was0 under the installed API's view conversion;
the new arithmetic itself was metered. Arena setup was separately billed.
The old unexecuted base-NumPy checker remains preserved alongside its
prelaunch ingress repair and denominator acceptance note.

Authoritative result synthetic/result.json SHA256
1e2f1a08b852fa9378341012e5ba5ddf7246b63044794b44af96151771924503.
Executed checker check_synthetic_v2.py SHA256
3f83518ba6462caa2664827f1be4d6c16b2e2d6af894bf0c44367889fca249cf.
All bound source hashes were rechecked unchanged after failure; completion
seal agrees and contains passed=false. Worker exited1 and is no longer
running. Swap unchanged776.75MiB of2048MiB; no MPS job.

Close this fixed symmetrize-plus-resident512 compiler and its execution
arm. No block/depth/precision/tolerance variation, candidate build, raw
screen, panel, package or submission follows this failure. This is a
specific numerical prerequisite failure, not proof that all symmetric
congruence algorithms fail or that the physical covariance is indefinite.

Official329926 remains4.982231460183304e-9; qualified local0c21 remains
4.7768418485455745e-9. Its initial/cold Gram qualification is separate
and unchanged. Preserve all frozen work. The <4/3/2e-9 milestones and
OFFICIAL strictly<1e-9 final goal remain unmet and active.
