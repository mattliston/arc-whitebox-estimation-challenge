# Fixed PUSH 3+3+1 microbatch kernel comparison — source draft

This is a prospective timeout-repair component check, not an estimator, score,
candidate build, full-card qualification, upload or release. Source preparation
does not authorize execution. All prior estimator, kernel, timing and negative
official receipts remain unchanged. In particular, local runtime evidence did
not prevent current c2 submission #330840 from failing the official timeout.

## One numerical question

For only `(m,k,n)=(1024,384,8192)`, can three fixed independent-child batches
`(3,3,1)` reduce the current serial-top D6 PUSH kernel's RPC wall time while
preserving its complete FP32 output bytes and warm FLOPs? No shape, depth,
batch-size, layout, precision, backend, seed or parameter search is allowed.

The fixed timing usefulness threshold in this draft is
`max(micro3 warm call1,call2)/max(serialD6 warm call1,call2) <= 0.90`.
Operational success is separate from that Boolean. A false timing result is
retained without retry or retuning. D4 is an original timing/error reference,
not an alternative selected or banked by this component test. Root must review
this entire prospective contract before selecting any execution.

## Source and physical ownership

The exact baseline is `../current_comp_shared_gain_slice_v1/build_v1/estimator.py`,
SHA `c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea`.
`kernel.source_bundle()` authenticates that file without importing it and
extracts only its last-effective binding class, view binder, packed frame,
packed child product, serial D6 product, and the six relevant layout literals.
It never executes the estimator, imports whestbench or loads model literals.

The D4 and serial D6 arithmetic are those exact source spans. The micro3 frame,
child and top functions are unique, positional, whole-span inverses of the
same original definitions. Changes are limited to function names, an explicit
batch dimension, basic view slicing, and the fixed three-group child loop.
All original forward transforms, leaf matmul and reconstruction statements
retain their operation and operand order. The original READ kernel is not
called or changed; both its original and dormant D6 view bindings are retained
where current serial D6 retained them. No scalar, physical term or head changes.

The batch dimension is immediately before the two matrix axes, after the
recursive seven axes: `(7,)*level + (batch, rows, columns)`. `unstack(axis=0)`
still unpacks the original seven Winograd products. The top inputs and output
products use only basic views `[0:3]`, `[3:6]`, `[6:7]`. The final single batch
uses one lane of the same triple child storage. There is no padding, transpose,
new arithmetic, output reconstruction after the original top reconstruction,
or new backing array. Batch matmul dispatch may nevertheless change actual
floating-point execution: exact byte equality is an experimental gate, not a
source-level promise.

The unchanged arena is 292,225,024 FP32 elements / 1,168,900,096 bytes. Original
top storage is 37,486,592 elements; one serial child needs 80,519,936 elements.
Triple child storage ends at 279,046,400 elements / 1,116,185,600 bytes, within
that arena. Interval scaling retains the original disjoint/live-overlap plan
in each lane. The 34 new PUSH views replace the 34 original PUSH views; they do
not supplement them. Existing original D4 and READ binding charges remain.

## Exact complete component cost

The independent original price owner is
`../current_d50_serial_d6_kernel_v1/kernel.py`, SHA
`a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7`.
Its fixed-depth complete arithmetic/copy tariff is rechecked against:

`7^d (m/2^d)(n/2^d)(2k/2^d-1)
 + (7/4)(mk+kn+mn) sum_{j=0}^{d-1}(7/4)^j`.

| Fixed role | Warm product F | Numerical calls | Cold zero-F unstack calls | Complete setup F |
| --- | ---: | ---: | ---: | ---: |
| Original D4 | 3,930,963,456 | 73 | 16 | 1,068,953,600 |
| Current serial D6 | 3,421,951,456 | 655 | 24 | 1,630,009,600 |
| Micro3 D6 | 3,421,951,456 | 291 | 44 | 2,112,169,728 |

Numerical call counts `(add,subtract,stack,copyto,matmul)` are respectively
`(28,32,8,4,1)`, `(252,288,72,36,7)` and `(112,128,32,16,3)`.
Original top18 plus three unchanged child91 calls gives291; seven leaf products
are still performed, now in three matmul dispatches. Cold frame discovery is
counted separately, never silently rebased to a warm operation count.

The original D4 PUSH+READ reshape sum is 1,068,953,600 F. Serial D6 adds
278,566,656 PUSH + 282,489,344 READ F. Micro3 replaces PUSH with 760,726,784 F:
**482,160,128 additional cold F**, no additional warm F. Setup includes actual
input `fnp.load`, one arena/output allocation and every binding in a paid
context. In the pinned implementation loads/empty allocations charge zero;
the actual whole bill must equal the complete stated reshape tariff. Nothing
is credited away because a view does not allocate memory.
Setup also requires exactly two zero-F `empty` calls and respectively48/116/116
paid `reshape` calls; the original NPY loads do not add operation rows.

Output `_fetch_data` uses a distinct actual paid materialization context and
its observed F0 plus wall/backend/overhead/residual are retained. Thus inclusive
setup+first-product F are D4 4,999,917,056; serial D6 5,051,961,056; micro3
5,534,121,184. Subsequent complete product+materialization F equal warm F.
These are component bills, not complete estimator costs. Input fixture creation,
FP64 synthetic reference and post-retirement diagnostics are preparation/judge
work, explicitly outside the component F but inside the whole observed runtime
and memory scope. No full-card mask, INIT or runtime is borrowed from c2.

## Fixed inputs, call roster and gates

`worker.py --mode prepare` creates only its own synthetic files. Generator is
NumPy `Generator(PCG64(9309094))`; left is
`standard_normal((1024,384), dtype=float32)/float32(sqrt(384))`; right is the next
`standard_normal((384,8192), dtype=float32)` draw. The independent reference is
`left.astype(float64) @ right.astype(float64)`. No existing arrays, graph data,
weights or truths are used. Three exact NPY file/payload hashes, shapes and
dtypes are sealed before any timed role. No seed tuning or fixture retry.

Role order is fixed **D4, serialD6, microbatch3**. Each gets a fresh local IPC
server/client lifetime, exactly one setup and three products (cold0,warm1,warm2)
using the same arrays. Each product has separate materialization. This is nine
products and 21 BudgetContexts, with 12 retired setup/product groups. Arena and
inputs intentionally live across a role's products; frame caches/binding state
are cleared and all four persistent array weakrefs must die before that client
publishes its positive three seals. Both process groups are reaped before the
next role. There are no interface repeats beyond this exact roster.

Every complete setup, core and fetch bill must be finite, use fewer than2^39 F,
take less than60s, and report RPC residual at most0.4s. Actual operation tariffs
must sum to complete F; exact numerical and cold-view counts must match the
table. No warm extra operations are accepted. Every role's three outputs must
be byte-identical; all three serial/micro outputs must also be byte-identical.
Actual signed kernel outputs must be finite, with exact FP32 shape and full
33,554,432-byte wire/save lengths. Negative matrix entries are valid here:
this is not a nonnegative physical estimator output. D4 and FP64 max/RMS errors
are reported without converting them into a new accuracy threshold.

Judge admission authenticates all three positive role three-seal chains,
complete bills, all saved outputs and fixture/source maps before decoding its
own synthetic arrays. It reports the single prospective timing ratio and
`component_promising`; it does not require that Boolean for operational PASS.
Nothing automatically advances, builds a candidate, scores a target or fixes
the official timeout.

## Backend, interfaces and exclusive outputs

All files are inert without explicit `--run`. The server and synthetic
prepare/judge use the pinned production OpenBLAS environment; the client uses
the real NumPy-free `.client-venv` and unchanged local IPC flopscope server.
`../current_comp_shared_pair256_v1/tiny_worker.py`
`c6afc327c9acb6a9cd6ba3ed66ef293f8a05d9a22e9031b97ffe9cec49d42e1d`
is reused **only** for its reviewed backend origins/BLAS attestation and
numerical source pins. It grants no pair-metric or full-model eligibility.
The original kernel client source is also preserved/pinned as the actual
setup/core/materialization lifecycle reference. No ancestor scorer or panel
admission is reconstructed.

Launch only through `watch.py --run --numerical-slot-clear` with six required
actual source SHA flags: `--source-sha256` (watch), `--worker-sha256`,
`--kernel-sha256`, `--client-sha256`, `--protocol-sha256`, `--spec-sha256`.
The watcher supplies worker flags `--source-sha256` (worker), `--kernel-sha256`,
`--client-sha256`, `--watcher-sha256`, `--protocol-sha256`,
`--watch-spec-sha256`, `--mode prepare|server|remote|judge`; roles and actual
new `--contract-sha256`/owned IPC URL are added where applicable. Client is a
hash-first explicit `--run` entry, forwarding the same remote arguments.

Exclusive outputs: `run_v1/prepare`, each `run_v1/{role}/client`,
`run_v1/{role}/server_attestation.json`, `run_v1/judge`, and `watch_v1`.
Prepare/client/judge each publish actual result/completion/finalization seals;
server authority is its attestation plus externally reaped process record.
Worker schema is `current_comp_push_microbatch3_kernel_v1`, distinguished by
mode/role. Watch schema is `current_comp_push_microbatch3_kernel_watch_v1`.
Existing output directories fail before any publication; there is no resume.

Whole runtime is300s, actual full tree strictly below8GB, zero added system
swap, every completed sample at most1s, at most120s since an actual retired
context group; every worker/server lifetime is also bounded at120s. Six BLAS
variables are1, nice10, assertions on, bytecodeoff, fallback0, no PYTHONPATH,
fixed4GiB maximum individual array. The final complete inventory, all cleanup,
logs and three watch seals share those limits and the final one-second
publication tail. See WATCH_SPEC.md. No sources or stage have been executed by
the source author; source review, exact inverse/API preflight and a separate
root launch decision are still required.
