# Prior batching work reviewed before the current reprice

The earlier global cross-subproblem Smirnov batching study actually ran;
it is not an untested optimization. Its representative product1024x832
by832x8192 lowered counted warm calls3623 to1083, but warm1.021447375s
was8.0960 times the contemporaneous D4 baseline. Its batched matrix
multiplications alone took.293353285s, exceeding the entire allowed
warmwall.151399950s. Eliminating Python dispatch alone cannot rescue
that particular schedule. Historical root MPS activity was concurrent, so
those are local same-run ratios, not isolated grader measurements.

That executable had signed Smirnov linear circuits52/111/146, delayed output
writes, a static largest-first arena and global cross-subproblem batching.
Byte parity with its frozen Smirnov control passed; relative FP64 Frobenius
error1.4472458212292093e-4 passed its2e-4 gate. Its runtime gate failed.
The packed arena/padding retained1162085136B; peak process1647820800B.
This is neither a broad Smirnov impossibility proof nor current-Lazy runtime.

The separate dense-grouped preflight also ran. Dense transforms replaced
sparse52/111/146 work by680/1400/1422 per block position. Three levels
priced11.420613624GF in transforms alone. Grouping independent products
with block-diagonal zero-filled dense matrices cost9.47264/9.634368GF,
already above the old entire D4 bill. Zero entries do not discount billed
dense arithmetic. Such packing is not an unclaimed saving.

Within-linear-DAG same-depth sparse batching with gather/scatter is a
distinct possible implementation, not present in the inspected source.
It introduces copying/workspace and by itself leaves the demonstrated
small-leaf-matmul bottleneck unresolved. No new sparse-batching kernel or
benchmark was selected from this review. The current scalar reprice first
tests the finite family's residual arithmetic scope on accepted Lazy, and
fails its necessary standalone3e-9 gate.

Pinned sources: SMIRNOV_GLOBAL_BATCH_REPRESENTATIVE_V1_CLOSED.md,
DENSE_GROUPED_FMM_COST_PREFLIGHT_2026_09_07_V1.md, executable transpose
certificate and old envelope scripts in astra_phase2_reexamination_v1.
No old benchmark was rerun and no installed library was modified.
