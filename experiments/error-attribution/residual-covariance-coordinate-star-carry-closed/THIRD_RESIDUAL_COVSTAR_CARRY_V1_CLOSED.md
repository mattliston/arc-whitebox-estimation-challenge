# Residual covariance/coordinate-star carry — CLOSED

2026-09-06 04:59:21PDT, Phase2. Single admitted training mini57, pre13 ->
pre14 -> isolated mean15 response. Full parent retained; only oracle
off-diagonal K21 ERROR completed and transported through Gaussian gates.

Dense small tensor/affine certificates passed<=1.11e-16. All18 original
pre13/pre14 input and prefix checks were byte-exact; pre15 response comes
from the actual scalar-final path. Source repeated-plane relative error
1.25e-16. New pre14 target checks pass: affine mean1.01e-10 RMS, covariance
2.74e-6 relative, stored-f32 rounding bound .0003144 of missing-plane RMS.
No clipping, fit, new samples or additional public file.

Natural oracle carry leaves covariance-response error ratio .8802095124
and next-mean response ratio .8686113339 (both required<=.75). Cosines
.35736/.36262. Weighted K21 error ratio .8371206130. Source error norm
1.71942; propagated scalar marginal was excluded as designed. FAIL.

This closes this fixed residual completion/linear-carry route, not all
residual tensors or a new whole-state replacement. No source learner,
alternate completion, strength/frame/cell sweep, recurrent sidecar, panel
or submission follows. Ratios are single-stream oracle consumer errors,
not final raw MSE, adjusted scores, or independent-noise ceilings.
Receipt third_residual_covstar_carry_v1.json/.npz; helper SHA256
04c37d7c4e8f04b05f4c73c0061bdda6db1c7f129f6b0321713d3f08c68b5094.
