# Fixed native-G0 forward test: FAILED

2026-09-09 UTC / September 8 Pacific. Phase 2 only. This attempt is frozen.

Six CPU forward calls were made on TRAIN35: disabled null, then enabled null,
plus025, minus025, plus050, minus050. The disabled adapter reproduced all16
old CPU output bytes and all47 ordered selector records. All five enabled
layer0 outputs matched the native bank byte-for-byte. All16 maximum absolute
differences from their respective native outputs were 2.14577e-6 to2.86102e-6,
inside the unchanged 1e-5 forward bound.

The first scalar-response comparison FAILED:

- Rank35, fixed amplitude0.025, original CPU64 scalar/cotangent.
- Native symmetric secant: -4.3437089166653564e-5.
- Native-G0 CPU symmetric secant: -4.0254010785778505e-5.
- Relative discrepancy: 0.07328019538009681, above the fixed0.05 limit.

The sequence stopped before the0.05 scalar verdict and every rank100 call.
No gradients, targets, training, model updates, new native scoring or board
submission occurred. The older unseeded bridge remains failed. This result
does not authorize the frozen CPU pilot or a threshold/amplitude rescue.

What it establishes: exact native initial Gram removes the observed layer0
mean mismatch, but that isolated change is insufficient for response fidelity.
It does not identify which later covariance, carrier, pair-map or head
arithmetic explains the remaining discrepancy, nor rule out adaptation's
potential value. No score gain was measured; bank #330390 remains official
4.53886349950395e-9 adjusted, raw2.120838555441651e-8.

Child elapsed50.6093s; independent watchdog51.3637s, max sampled RSS
5,822,005,248 bytes, zero swap growth, no resource termination. Both completion
seals were verified. A separate read-only post-failure check verified every
recorded source, input and artifact hash; this does not relabel the failed
child's incomplete acceptance sequence.

Receipts (SHA256):

- forward_v1/result.json: 7794394632f3b96866d1e539753700699c53e37734aa5d2455b9f62b6c7ef1b3
- watch_v1/result.json: 5d230ee2ab2bb494157e14ee690294f0091c6828881f10a3093bac4fb976b712

The first launch's permission reviewer timed out before process creation.
The single allowed retry launched this one numerical attempt; there was no
numerical retry.
