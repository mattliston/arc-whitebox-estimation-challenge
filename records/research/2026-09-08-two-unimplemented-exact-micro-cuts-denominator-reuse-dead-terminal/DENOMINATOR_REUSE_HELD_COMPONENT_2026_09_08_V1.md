# Unimplemented exact scalar reuse: retain for a combined change

September 8, 2026 PDT. Phase 2 only. Root and the existing efficiency agent
inspected the current qualified source and installed tariffs. No estimator,
array acquisition, new candidate, complete bill, panel or runtime run.

Reference: unused_materializations_v1/estimator.py, SHA256
409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f.
The parent is frozen and was not edited.

In _nonlinear_pair_batch, the covariance-metric denominator of its returned
POST covariance (named cov inside the map and C in the caller)

    trace(C)^2 + 2*sum(C*C)

is computed at the end of the pair map. The effective _predict_binary
recomputes it for omitted_radial_source at layers 1..14, before the head
changes C. The intervening host copy and omitted_radial_source do not mutate
C. The already computed scalar could be returned in aux and reused there.
No full matrix cache, changed reduction order or changed formula is needed.
Layer0 has no previous CP bank; terminal15 does not execute this restoration.

Installed FNP.12 pricing for the repeated FP64 expression is:

- trace: 2*n;
- scalar power2: 32;
- C*C: 2*n^2;
- reduction: 2*(n^2-1);
- final scalar multiply and add: 4.

Total: 4*n^2+2*n+34 per occurrence, or 58,749,404 F for fourteen n1024
occurrences. This is 0.01260558% of the qualified mean466,058,731,667.9286F.
If source/byte parity and that complete delta are later verified on every
panel net, expected mean F is465,999,982,263.9286 and adjusted would be
4.769635430744964e-9 rather than4.770235434670105e-9. The latter arithmetic
subtracts deltaF/2^41 times the same panel's mean raw because deltaF is
constant and every multiplier is above the floor; it does not multiply
independently averaged raw and cost. These are CONDITIONAL source-price
projections, not new metered scores or a runtime forecast.

No standalone qualification run is selected for this tiny lead. Preserve
it for an otherwise justified combined implementation, with a fresh fixed
protocol, exact all16 byte tests, complete bills and release/resource checks.
Do not call it a validated component or add its gain to the qualified score.
Previously recorded overwritten-radial work and the four already-qualified
materialization cleanups are not counted again here.

The no-refit response-row exception remains complete, frozen work remains
unchanged, and there is no milestone submission. Official strictly<1e-9,
with qualified banks below4/3/2e-9, is still the active goal.
