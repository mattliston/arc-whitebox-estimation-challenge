# One fixed representative global-batch test

September6,2026 PDT. Root authorized exactly this test after static v2
reproduced all frozen bills/lifetimes and verified the new concrete packing.
No MPS, target data, algorithm/group/shape/thread/backend sweep, candidate,
full-panel run or submission. Root's MPS training remains concurrent and
will be explicitly labelled; timings are not isolated service timings.

Use native official FNP0.12, one BLAS thread, geometry1024x832 times832x8192,
seed930066 with the original small-fixture draws consumed first, preserving
the prior full fixture. The frozen primary3x3x6x3 / Winograd schedule is
unchanged. The global schedule is exactly static v2: contiguous batch-axis
streaming of12800 leaves, deferred ancestor outputs, immediate per-side
input release, late odd-k fringe, verified largest-first addresses.

Use one shared scratch array for the D4 reference, the old frozen Smirnov
numeric reference and the new schedule; calls are sequential. Retain only
the padding needed by the currently tested Smirnov object. Shape-only view
binding work is counted on its first product call, not moved outside the
measurement. All arithmetic/copies remain ordinary FNP operations inside
BudgetContext. Reference input/output/NumPy arithmetic is external test
harness work, never called by the tested product.

One cold and one warm D4 call; one frozen Smirnov call for numerical
reference; then one cold and one warm global-batch call. No retry or sweep.
Required prerequisites:

- New outputs finite and <=2e-4 relative Frobenius error versus f64 NumPy.
- New output relative difference <=1e-10 versus the frozen arithmetic
  program. Reordering independent batch elements should preserve each
  arithmetic path; exact output hashes are reported too.
- New cold/warm FLOPs exactly7,258,335,196 /6,611,249,012. D4 exactly
  8,417,538,304; frozen Smirnov exactly6,834,384,092 on its cold call.
- New retained work<=1,168,900,096B; planned arena1,115,801,600B and
  padding46,283,536B. Every allocation/release must match the static trace.
- Whole benchmark process peak RSS<=2,000,000,000B.
- New WARM wall <=1.20x the same-run warm D4 wall; COLD <=1.50x cold D4.

The wall gates are bank-relevant necessary conditions, stricter than the
old3x prototype gate. Passing is not deployment approval: actual CP inputs,
full-card raw/bill, both panels, and persistent total/residual wall plus
memory would require separate coordination and verification. Failing
preserves this receipt and stops without a shape/batch/cache/leaf sweep.

