# Unchanged D4 baseline: transforms dominate these two synthetic shapes

September 8, 2026 PDT. Phase 2 only. No score improvement, changed kernel,
MLP evaluation, deeper schedule, release or candidate qualification.

The fixed protocol completed normally in one nice-10 OFFPY process,
PID95540, 11:12:16.039695--11:12:17.196459 UTC. `run()` wall was
1.15677112527s, excluding interpreter/module imports; peak process RSS
1,312,374,784 bytes. There was no MPS or remote server. No BLAS pool was
discoverable: the three one-thread environment guards are attested, NOT
an observed native BLAS thread pool. The version/prefix guards passed.

The three existing source functions were compiled from their unchanged
AST bodies. Both synthetic shapes used current maximum-capacity views,
depth4, seed52907, one cold invocation and two warm invocations. There
were no model weights, labels or payloads. Inputs/setup and output-byte
inspection were outside the kernel context. Setup was separately billed.

| Shape (m,k,n) | Kernel F, each call | Separate setup F | Warm backend seconds, both calls | Matmul fraction, both calls |
| --- | ---: | ---: | --- | --- |
|1024,832,8192|8,417,538,304|524,226,560|0.1128722113 / 0.1065244591|17.0538357931% / 15.6864989942%|
|832,8192,832|6,921,666,752|544,727,040|0.1210212931 / 0.1202292107|14.6676666850% / 14.3377180171%|

For every invocation the primitive counts were 28 add, 32 subtract,
8 stack, 4 copyto and 1 matmul. Cold calls additionally bound public
unstack views at zero F. The operation totals equal each public bill and
the predetermined warm formula. All three output hashes agree within
each shape and are finite FP32. Setup had 24 billed reshapes per shape.
All source/protocol/layout/runner pins remained unchanged at exit.

Thus add/subtract/stack/copyto account for 82.9462--85.6623% of measured
warm backend time on THESE TWO fixed native shapes. This identifies a
runtime target, not measured physical DRAM traffic or a proof of bandwidth
saturation. It does not estimate whole-estimator shares, remote/official
time, a serial-top or deeper-kernel gain, raw error or adjusted score.
Any changed implementation needs its own prospective admission and all
same-configuration numerical, full-cost, resource and release checks.

The existing efficiency agent independently checked every operation sum,
formula, fraction and pin against the completed receipt. Root rehashed:

- result.json: b4081393d022ffcd5a6996ad18f7152275e7873eda0cfcded6a3dc4ca5fd197b
- run_profile.py: 65e0f634db924de7f1b54836c2bef07d179e590645f897df017811fd7823a7d8
- PROTOCOL.md: 9e24466e4716b9c031a61a403ffea607fc48f37e23bee06a4fff967250770a7a
- qualified estimator: 409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f
- packed_arena_generic.py: 4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e

Official329926 remains4.982231460183304e-9 adjusted. Qualified local14
remains4.770235434670105e-9. No milestone or upload was earned.
