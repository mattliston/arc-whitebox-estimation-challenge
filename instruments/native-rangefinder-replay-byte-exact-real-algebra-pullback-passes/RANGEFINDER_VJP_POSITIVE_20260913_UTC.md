# Native-forward real-algebra VJP: fixed qualifier passed

All three predeclared cases passed the unchanged absolute1e-8 AND relative1e-5
dot criteria. On actual ENTRY8, absolute error1.2131779914291213e-10 and relative
error2.577197043039642e-6. All five independent production leaf gradients were
finite/nonzero. Every checkpoint gradient and output byte matched the plain
run; reconstructed and saved native Y bytes matched. Actual20 native products
included checkpoint recomputation. No full-card calls, targets, parameter
updates, finite differences, tuning or score measurement.

Whole worker5.060914000496268s, independent watcher5.356660791672766s;
peak combined RSS1415938048 bytes, maximum completed observation interval
.2489666659384966s, zero added swap. All owners and worker group retired;
session45798 reaped with exit0. Worker result/completion/finalization seals:
55eec7f39a7c17af1e65c89b72d22b38b22bb40508db9e2acc6297ff2610e805,
5502f31a7aafc96ccb4f854d44875684aa24ffd109cb69195321a12bf904f368,
d390c453399aa868ecb0d3fbeeb08a2837a7e5cd5658f65a2f913c56724246be.
Exact independent watcher seals are pinned in rangefinder_full_inputs.json.

This qualifies the intended real-algebra derivative at rounded intermediates,
not differentiation of rounding, full-model gradients, a fit or release.
Next permitted question is one full canonical35 forward with the same
native-backed Y at all ten original joins and all other model/head arithmetic
unchanged. The existing all16 maxabs1e-5 fidelity gate stays in force. No bank
has improved: official #330390 remains4.53886349950395e-9.
