# Fixed message/covariance CPU training-engine qualification

Prerequisite: native integration qualification passed all10 full calls and all
six guards, with all controller/worker groups retired. Consume its exact zero
source; official331760's P/A/cstar/old edge/scales stay frozen. Only two
48x16 FP32 matrices(context_left/context_right) may require gradients.
Model initialization always reads zero context, never the native sentinel.

Unit: the three existing synthetic native fixtures, at zero and fixed nonzero
context (six cells). Old zero-cell output bytes must match. Independent FP64
assembly/pullback after explicit FP32 normalization/message boundaries:
forward relative L2<=2e-6; each context-gradient relative L2<=2e-5;
analytic directional derivative versus central finite difference<=2e-6,
step1e-5, denominator floor1e-12. All old leaves stay unchanged and gradient
free. No Monte Carlo/network targets are read.

Full CPU calls in fixed order: frozen35,checkpointed35,checkpointed100,
checkpointed35. Three synthetic final-output directional backwards use the
existing cosine direction and1000 scale. Require exact frozen/checkpoint
outputs,35 repeated gradients/domain/selectors/state traces, all finite
nonzero context gradients and all old values unchanged. The first frozen35
must exactly reproduce the authenticated old frozen CPU engine's16 outputs,
traces and old parameter bytes. This is CPU-to-CPU zero parity, not full native
forward fidelity or a native derivative claim. Global Torch seed9409214.

Use the unchanged real Adam state/RNG boundary helper. One synthetic update,
save/restore into fresh leaves and optimizer, then one identical synthetic
update must preserve exact tensors/settings/RNG. Saved qualification states
are ineligible for candidate initialization. Full-fit process scheduling and
terminal selection must be declared separately before any fit targets.

The underlying ordinary-product/FP64-QR CPU engine remains an approximate
proposal engine. These checks establish component arithmetic and repeatable
optimizer transport, not the derivative of the native FMM estimator.
No accuracy score, fitted model, native export, release or upload follows
automatically. Eventual native40/downstream/client gates remain unchanged.

One serial nice10 CPU worker, six BLAS limits1,RSS<7.5GB,zero added swap;
retired Card RSS<2GB, same allocator ABI helper, .05s guard polling, emergency
256MiB added swap cutoff, disk>=470GB, artifacts<100MB. Unit180s/full1200s.
No retry, samples, downloads or GPU. All sources and fixtures are sealed first.
