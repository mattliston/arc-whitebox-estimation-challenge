# Restricted D4 views: numerical consistency passes, timing gate fails

September8,2026 PDT. Phase2 only. Completed normally11:41:00.439245--
11:41:02.390199 UTC, nice10 PID837. The fixed two-shape diagnostic is
CLOSED under its unchanged prospective gate. No candidate, depth retry,
MLP/target access, fit, remote server, panel or submission followed.

All12 kernel calls were finite FP32 and byte-identical within/across the
two layouts for each shape, including calls alternating on the shared
original arena. The original source AST bodies and numerical operations
were unchanged. Every call retained28 add,32 subtract,8 stack,4 copyto,
1 matmul and the predetermined complete KERNEL bill. All public-operation
totals and separately billed setup counts reconciled.

The fixed warm order was capacity,restricted,restricted,capacity. These
are means of both warm calls per configuration; no best repeat selection:

| Shape (m,k,n) | Capacity outer seconds | Restricted outer seconds | Outer ratio | Backend ratio |
| --- | ---: | ---: | ---: | ---: |
|1024,832,8192|.10950174974277616|.11568529112264514|1.0564697951803907|1.0588303637049927|
|832,8192,832|.1277810619212687|.12281343713402748|.9611239356399934|.9610505601097808|

Both shapes fail the required <=.90 on both timing ratios. This is a
fixed short native result, not a statistical or universal runtime theorem.
Cold timings are preserved but not used to select a winner; both layouts
share an arena and the capacity cold call came first.

Kernel F remains8,417,538,304 push /6,921,666,752 read. Separate24-reshape
bindings cost469,863,680 /438,129,120F restricted, versus524,226,560 /
544,727,040F capacity. This confirms a per-binding price, NOT a lower
complete estimator cost: a deployment would need each unseen shape's
binding/cache policy, replacing the current one-time bindings. No warm
FLOP saving, changed raw mean, adjusted gain or validated release component
is established by these two fixtures. The safe interval-subset construction
and exact-byte facts are retained, but no timing/order/shape/depth sweep
or full-card run follows from the failed admission gate.

`run()` wall1.9509516665712s excludes interpreter/module imports and
includes fixtures, all contexts and output inspection. Peak process RSS
1,364,000,768B. Native threadpool introspection was empty, so only the
three one-thread ENV guards are attested. Version/prefix/nice guards and
source/protocol/layout/runner exit hashes passed. Post-run process-name
check found no Python/Whest/Torch/FNP worker; swap776.75MiB/2048MiB,
unchanged from before execution. No MPS job was launched.

Independent efficiency-agent receipt review PASS, root rehashed:

- result.json: 0d6cac70893589d1286cd2540356ba475b4b9a8839ad290908e9b2f9dcc65695
- run_compare.py: f1b4cd347f9c725d8220a5736a256ca836dba9e2652cf5252a947e89e4dacbc8
- PROTOCOL.md: c1ae5ca583a17c673eac19dfe814bf0fd23e8364385c8dd23567abc6d2076755
- unchanged qualified estimator: 409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f
- unchanged layout: 4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e

Official329926 remains4.982231460183304e-9 adjusted; qualified local14
remains4.770235434670105e-9. The strict official<1e-9 goal remains active
and unmet. No <4/3/2e-9 milestone is earned. No forum, commit or promotion.
