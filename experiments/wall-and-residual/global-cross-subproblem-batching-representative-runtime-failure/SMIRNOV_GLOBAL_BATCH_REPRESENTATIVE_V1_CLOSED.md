# Global cross-subproblem batching: representative runtime failure

September 6, 2026, 11:55:12 PDT (receipt finish and filesystem mtime agree).
The authorized single fixed test completed normally, exec session 22388,
exit 0. No estimator, panel, submission, MPS call, or subsequent sweep.

## Verdict

**CLOSED as a bank-cost route in this schedule.** Cross-subproblem batching
preserved the old arithmetic exactly and reduced counted warm calls from
3,623 to 1,083, but failed both preregistered bank-relative wall gates.
Static fit and lower operation counts did not predict usable runtime.

Fixture: deterministic seed 930066, including the frozen small-fixture
draws before the representative A(1024,832) x B(832,8192) inputs. Native
official flopscope 0.12, one BLAS thread, escalated nice 10, one shared
scratch buffer, no target data. Root MPS training was concurrent, so these
are local same-run prerequisite timings, not isolated grader timings.

| Product | Cold seconds | Warm seconds | Cold FLOPs | Warm FLOPs |
| --- | ---: | ---: | ---: | ---: |
| Same-run bank D4 | 0.241809708 | 0.126166625 | 8,417,538,304 | 8,417,538,304 |
| Frozen Smirnov numerical reference | 1.317706667 | Not measured | 6,834,384,092 | Not measured |
| Global-batch schedule | 1.181486750 | 1.021447375 | 7,258,335,196 | 6,611,249,012 |

Global/D4 cold ratio **4.886018678**, versus gate <=1.50: FAIL.
Global/D4 warm ratio **8.096018863**, versus gate <=1.20: FAIL.
Warm kernel-only bill savings are 21.4586%; this is not a demonstrated
whole-bank saving. The larger cold shape-binding bill was correctly
metered, not moved outside BudgetContext.

## Passing prerequisites

- Global cold and warm outputs are byte-identical to the frozen Smirnov
  output: SHA256 `30ed1a5f975b393f13be5930e633a6b8985cd053bd7334b854e1d3751eccbaf7`.
  Relative difference is exactly 0, versus <=1e-10 gate.
- All outputs finite. Global/frozen relative Frobenius error to float64
  NumPy is 1.4472458212292093e-4, versus <=2e-4 gate. Bank D4 error on the
  same fixture is 6.667789418363796e-6.
- All five context bills equal the preregistered exact expected bills.
- Every runtime allocation/release matches the verified static lifetime
  trace. Arena 1,115,801,600 bytes, dedicated zero-padding 46,283,536;
  retained workspace **1,162,085,136** <= bank cap 1,168,900,096 bytes.
- Actual whole-worker peak RSS **1,647,820,800 bytes** <=2,000,000,000.

## Causal evidence and scope

The new mechanism really did reduce accounting/dispatch calls: warm
arithmetic counts are add 457, subtract 478, multiply 72 and matmul 40;
reshape 12 and copyto 24. The exact warm element/arithmetic bill is
unchanged from the frozen algorithm.

However, global warm BudgetContext splits are backend **0.904923955 s**,
flopscope overhead **0.097706693 s**, and residual **0.018941809 s**.
The actual backend alone already exceeds the entire permitted warm wall
(1.20 x 0.126166625 = 0.151399950 s) by nearly 6x. Its batched matmuls
alone take 0.293353285 s. Therefore simply eliminating all remaining host
dispatch/accounting overhead cannot rescue this tested numerical schedule.
This does not license attempts to bypass metering or blanket claims about
all fast-multiplication algorithms; it closes this fixed global-batching
schedule and leaves the prior prebinding-only and frozen-stream8 failures
unchanged. No shape, batch, leaf, cache, backend or threading sweep follows.

All earlier receipts remain untouched: static v1 zeros-accounting mismatch,
v2 first-fit memory failure, successful v2 largest-first placement, frozen
stream8 runtime failure, and the tiny prebinding-only diagnostic.

## Immutable receipts

- Design: `SMIRNOV_GLOBAL_BATCH_REPRESENTATIVE_DESIGN_V1.md`.
- Actual results: `smirnov_global_batch_benchmark_v1.json`.
- Harness: `benchmark_smirnov_global_batch_v1.py`, SHA256
  `55c45483a11def727e7204b2f3a5a7382d5f3a8131ab763c98a1b08513be758f`.
- Kernel: `smirnov_global_batch_native_v1.py`, SHA256
  `7c4f6b48af67e2032f0ba57bb2e47746b58ad1871967553c4a4760f1c378ac62`.
- Static prerequisites: `smirnov_global_batch_static_v2.json`.

No accuracy gain, accepted bank improvement, or official-score gain is
claimed. CPU worker is terminal and the resource window is released.
