# Streamed scratch NEGATIVE — 2026-09-13 UTC

Session29891 is fully reaped. Fixed one-context RPC test passed all numerical,
complete-bill, owner-retirement and resource gates, but FAILED both
prospective component timing gates. No retry, shape/depth/cache ladder,
threshold relaxation or full streamed port is selected.

Same prospectively fixed product-through-paid-snapshot/fetch/hash boundary:
canonical_first0.15282954182475805s; stream_first0.20504137501120567s;
first ratio1.3416344285473047 >0.90.
canonical_warm0.11378770787268877s; stream_warm0.17003529239445925s;
warm ratio1.4943203934180924 >0.85.
The tested allocation/free policy is slower despite exact bytes and bounded
live storage. Do not equate successful lifetime validation with runtime gain.

All four (384,8192) FP32 outputs and both own repeats are byte-identical:
28a8ff2f7884be2fbeed5c495d73f45410a6bca80116a7474b7c147994d02736.
Against the independent FP64 product of the actual serialized stored operands:
relative Frobenius4.840484508017727e-6 and maxabs5.026594776857962e-6.
Product1512549696F, each required snapshot9437183F, timed1521986879F.
Complete one-context bill11557673227F includes all bindings, RNG, preparation,
four products, snapshots, stored-operand reference and all diagnostics.

Context0.899200875s: backend0.558032079s, overhead0.296546326s,
residual0.04462247s. Whole6.940713250078261s (finalization6.941596374846995);
whole-tree1320714240B; maximum completed sample interval0.7777772499248385s;
zero added swap. All arrays/context, owned/observational groups retired;
source/artifact hashes unchanged. No model, truth, raw or score was measured.

Positive operational result/completion/finalization seals (NOT timing success):
remote:91b0841ff01364916513d8631f4db450eb6e71a15efbe2b2e11fc243661d84d2
remote:a90dd8354e3a85ba7bfac8017adfcab091e615cb5df2afc745fe5ad8c7e805e6
remote:e5aa50797025391e14b02b595d5a8897f9e7049c06ec87b07b4a0440d9e5a9f8
watch:a91c15e35598b1a43ae316d443cc1613e1733b295518d0109fdc0a8500afd798
watch:7deae5e5c422ae5a495038a0accc9a2676c305ca0c4bd621db2c40d28bed819b
watch:8d559db5bdc6956c070278380384eb2e0b940a063d46e1d03ae5614b9c4a651f

Confirmed board bank unchanged: #330390,4.53886349950395e-9.
Prior exact full canonical-box47.65868s component remains preserved,
but does not pass the cold45s gate. No submission follows this unit.

