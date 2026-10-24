# Exact sign/constant circuit audit

Phase 2 only. September 5, 2026, 20:32 PDT. No estimator changes.

The existing primary Smirnov 3x3x6 rank40 shared straight-line program has
62/123/149 scalar linear operations. Audit a different exact circuit by
carrying rational scales and signs on symbolic wires, combining equal
scales before an addition, reusing identical operations, and eliminating
dead temporaries. Fold final A/B wire scales into the corresponding output
product wires. Every actual nonunit arithmetic operation is charged;
assignment/copy/live-memory costs remain explicitly outside this audit.

Certify the resulting exact rational bilinear tensor against every basis
product before repricing any matrix shape. This is not permission to use
unmetered sign manipulation or an alternative-basis rank count without its
transforms. Runtime operations must eventually be ordinary billed FNP.

Gate: only a whole-bank optimistic saving exceeding 25% on both existing
shape censuses, with necessary leaf storage within the existing arena,
justifies a subsequent numerical/memory/runtime kernel design. The prior
17.5% envelope alone remains closed for production implementation.
