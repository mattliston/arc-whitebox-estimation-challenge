# Fixed native bridge for the already sealed current-P perturbations

Prospective,2026-09-09 UTC / September8 Pacific. Phase2 only. This tests
implementation compatibility, not accuracy or training. No truth arrays,
optimizer, parameter update, fit, checkpoint search or board submission.

## Why this test is next

The explicit CPU64-moment/MPS32-CP hybrid passed all fixed forward/secant and
complete-P VJP checks on TRAIN35/100, with byte-identical CPU results and
all external watchdogs passing. Final hybrid receipt
f821503ed45372d6b9a14b363914ab255d643b4ae2d90b3eacb389f61d6892ec.
That does not prove compatibility with native d50 under changed P. This is
the cheapest remaining bridge before considering autonomous P adaptation.

The two-route reasoning in current_d50_hybrid_cp_qualification_v1/PROTOCOL.md
still applies: same-architecture adaptation has unknown raw gain and no added
inference operations at fixed geometry; difference-leg compression only has
an optimistic historical~4% cost opportunity and no current endpoint evidence.
The bank stays330390 at4.53886349950395e-9 adjusted. At unchanged bills,
about11.9% lower raw would cross4e-9. No such gain is forecast by this test.
An eventual changed-P candidate must measure its OWN complete bill and raw
error together; changed masks may change that bill. Final goal remains<1e-9.

## Exact five sources and fixed call order

Use d50 SHA d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451
and the actual FP32 P payloads already sealed in the old passed CPU-forward
receipt6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607.
Do not regenerate directions, change amplitudes, or fit anything.

Cases, in order: null,plus025,minus025,plus050,minus050. Null source remains
byte-identical d50. Each other source changes ONLY _GNN_LITERAL, using exact
stored array values. Check all keys/shapes/dtypes and raw bytes, unchanged
non-P fields, inactive layer0 head/bias/gain, exact cstar and source inverse.
No runtime payload loader or extra estimator arithmetic.

Each case has one fresh OFFPY process, executing the qualified native_v2
development loop35,100,repeat35. Thus10 unique case/net outputs and5 repeats.
All calls retain official owned FP32 FNP weights, separate input-copy budget,
initialization once, prediction/materialization, complete bill capture and
post-context diagnostic reset. Candidate state never resets between calls.
Adapt only candidate/provenance admission around the original numerical loop.
No CO downstream draft is imported or resumed. Every output directory is new.

## Preserved numerical, billing and resource gates

Require all hybrid stages AND their external watchdogs passed, frozen source
bindings unchanged, and no active numerical/MPS workers before acquisition.
The estimator is NumPy-free; the qualified OFFPY harness itself uses NumPy.
This is NOT remote NumPy-free-client service qualification.

One nice10 BLAS1 native worker at a time, no MPS. Existing per-context120s and
process8GB gates remain. Independent per-case watchdog900s, process/tree8GB,
stop on>2GiB swap growth. Preserve attempted output/bill before fail-stop.
No numerical/library environment changes. Complete FLOPs<=2^41 each call;
do not substitute a planned bill. Full original five-artifact-per-call
input/return/bill/lifecycle verification remains. Each35 repeat is byte-exact.
Initialization must be1383913786F for every variant; perturbed body bills may
change with geometry. Null35/100/repeat35 must reproduce the existing d50
panel_v2/terminal outputs and bills, receipt6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade.

After all outputs are sealed and verified, decode only those tiny outputs
and the old cotangents. Use the original CPU float64 scalar helper, including
the served FP32 output convention—not a replacement summation metric.
For every case/net require native-versus-HYBRID all16 max absolute difference
<=1e-5. For each rank and amplitude.025/.05 require native-versus-HYBRID
symmetric scalar secants agree within5%, denominator
max(abs(native),abs(hybrid),1e-10). Also report CPU comparisons separately;
agreement is not inferred transitively. No threshold/amplitude/seed rescue.

Any fidelity, billing, repeat or resource failure stops this bridge and is
preserved. No fit follows a failure. PASS establishes only this fixed native
bridge, not a useful raw gain or eligibility of arbitrary future P values.
Any later fit requires a separately prospective budget/terminal selection and
native complete adjusted evaluation, including fresh final source/head parity.
Official release retains8GB client+server,120s perMLP,.4s residual and local
<60s checks. No forum posts, commits, promotion or frozen-work overwrite.
