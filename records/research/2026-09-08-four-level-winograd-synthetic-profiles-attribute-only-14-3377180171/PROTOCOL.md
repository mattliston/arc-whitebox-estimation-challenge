# Unchanged current-D4 kernel operation profile

September8,2026 PDT, before execution. Phase2 only. This is a bounded,
target-free BASELINE diagnostic, not a new candidate, a D5/D6 retry, a
serial-top implementation, an accuracy test or a numerical release gate.

Question: within the unchanged qualified409f D4 helper, how much measured
backend time belongs to matmul versus add/subtract/stack/copyto? Saved
409f native bills retain only aggregate timing. The older D16 operation
split belongs to a1.272T card/3.580GB arena, and the later329884 client
profile collapses backend time into remote waits. Neither is a matching
current-kernel split. Do not infer bandwidth saturation solely from the
category names or transfer these native timings to the official worker.

Freeze source409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f.
Extract ONLY the existing bind/frame/product functions by AST, unchanged.
Use the existing packed_arena_generic.py layout, same maximum capacities
as the archived wall_headroom_kernel_v2 comparison. Do not insert timers
or namespaces into the kernel; collect the ordinary closed-context public
BudgetContext.summary_dict() operation totals after the call.
Layout SHA4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e
is also checked before and after the run.

Exactly two fixed synthetic FP32 shapes: push(1024,832,8192),
read(832,8192,832); input seed52907, one RNG in that order. No actual MLP,
weight, mean truth, target array or trained payload is read or imported.
Per shape: one cold-frame invocation followed by two warm invocations,
depth4 ONLY. Clear frame views between shapes. No best-repeat selection.
Check total operation bills equal their public sum, actual primitive
counts, finite FP32 outputs, exact repeat bytes and source stability.
This is profiling consistency, not a new floating-point error certificate.

Input generation, arena/view construction, and output hashing/inspection
are OUTSIDE the kernel BudgetContext and explicitly not in its bill.
Use the public setup context to record arena/view/input ingress separately.
Report all scope exclusions, run() wall (excluding interpreter/module
import startup) and peakRSS. No use of
this subtotal as a complete estimator bill or score. Warm helper arithmetic
must equal the predeclared formula

    7^4*(m/16)*(n/16)*(2*k/16-1)
       +(7/4)*(m*k+k*n+m*n)*sum_{j=0}^3(7/4)^j.

One nice10 OFFPY process, no MPS/server/network. Before numeric imports,
require the exact data_gate/.venv prefix, reject anaconda3, and require
NumPy2.4.6/flopscope0.12.0/Whest0.16.0/threadpoolctl3.6.0. Require
OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/VECLIB_MAXIMUM_THREADS all1, then
request one thread and check every discoverable pool. Current qualified
OFFPY native receipts have an empty pool list; if empty, explicitly
attest only the environment limits rather than claiming an observed pool.
Per-context budget2^38 and cooperative wall30s; overall run() wall<=60s
and RSS<4GB are POST-RUN consistency gates, not hard watchdog limits.
Failure produces a preserved failure receipt and no automatic repair/run.
No estimator source or frozen receipt is modified; outputs are exclusive
new files in this directory. No upload/forum/commit/promotion.

Decision rules: this diagnostic can identify the measured current kernel
categories worth investigating. It does NOT by itself admit a deeper
schedule, establish physical DRAM traffic, prove serial-top speedup, supply
an accuracy route or qualify any milestone. Those need their own concrete
mechanism and complete same-configuration evidence. No parameter sweep or
candidate follows automatically from any timing split.
