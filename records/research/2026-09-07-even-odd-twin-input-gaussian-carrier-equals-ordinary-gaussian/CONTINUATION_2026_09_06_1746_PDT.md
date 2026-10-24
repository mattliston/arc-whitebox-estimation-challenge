# Phase 2 continuation — September 6, 2026, 17:46 PDT

Clock checked as 2026-09-07 00:46:26 UTC. This is a new, append-only
checkpoint, following `CONTINUATION_2026_09_06_1704_PDT.md`. No frozen
package, existing user edit, or older handoff was changed. Standing goal:
official adjusted score strictly below 1e-9, banking validated crossings
below 4e-9, 3e-9 and 2e-9. Submissions are preauthorized; discussion-forum
posting is forbidden. Phase 2 only, at most two agents besides root.

## Board: unchanged, not a score-progress turn

Last confirmed usable bank remains #329926, official adjusted
4.982231460183304e-9 and raw final-layer MSE 2.2448352261505987e-8.
Evidence is the saved `SUBMISSION_329926_GRADED.md`, not a new live poll.
Exact per-network completion counts/timings were not supplied by that
status receipt. No candidate was built, panelled, metered or submitted
this turn. No crossing below 4e-9 has been banked.

At unchanged bank raw error the multiplier floor of 0.1 bottoms at
2.2448352261505987e-9. The strict 1e-9 goal therefore needs raw MSE below
1e-8 even at the floor, over 55% below the bank. None of this turn's
mathematical results supplies that improvement or a submission-ready lead.

## New source and mathematical results

1. `GAUSSIAN_RELU_SYMMETRY_SOURCE_AUDIT_V1.md`
   SHA256 `0bb03197c9012e084e2466774ff81929bf8786a8938abb1f09f13e6bae195b06`.
   A joint-Gaussian even/odd twin-input carrier collapses exactly to
   ordinary Gaussian mean/covariance propagation; the extra antithetic
   covariance does not add marginal information under that closure.
   Nonzero mixed even/odd third moments already appear after layer one.
   This rejects that Gaussian implementation, not all parity-based
   non-Gaussian states. No network experiment was run.

2. `COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md`
   SHA256 `0edcbe63aad1c534082e7192e459d007466411871b3facb146b2e02cd73ec5c2`.
   Audited the actual affine, antithetic, shifted-ridge, RQMC, Hermite,
   spectral and Stein prior-work scope. A deterministic bank constant
   cancels from an ordinary control-variate difference; it does not
   magically reduce sample variance. Zero-bias shallow ridge controls
   have only a linear odd part. Neither fact closes shifted/deeper
   controls in general. No new high-capture, correctly centered,
   affordable control was identified. No sampling or target access.

3. `GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md`
   SHA256 `8f8378be4aa44fe814b1ae6d00f38304857b95dcc9278820e23536f134790454`.
   The physical card's scalar radial fourth-order approximation is not
   invariant to arbitrary positive hidden-unit rescalings. This is NOT
   an automatic bug fix: its numerator and denominator match a ratio
   of Gaussian next-query ensemble expectations under an isotropic
   query metric. A gauge-covariant metric restores that interpretation;
   changing units while silently keeping the isotropic metric changes
   the projection objective. Learned heads are also not gauge-covariant.
   No normalization sweep, candidate edit or raw-score claim follows.

4. `SPHERICAL_STEIN_ONE_POTENTIAL_VARIANCE_AUDIT_V1.md`
   SHA256 `28811700f825cc262b28c4231c4c2f2a4c233a4f6ce3c439bba27038be9fd766`.
   Derived a genuinely distinct, exactly centered spherical Stein
   control with an all-output directional derivative recurrence. Its
   Gaussian projected-Hessian probes have a quantitative obstruction:
   even oracle potentials and an optimal radial extension leave about
   half of every nonlinear angular component with one probe. At the
   specified depth-16 dense circuit, independent probe averaging has
   nonlinear variance-times-cost at least 1.9374963009 times plain MC.
   This is not a universal lower bound for Stein methods, RQMC, fast
   multiplication, structured trace designs or an adapted Rademacher
   basis. The Haar-orientation Rademacher bound is only an average;
   it cannot be assigned to an individual optimized basis. No arbitrary
   smoothing-temperature, potential-width or probe-count run is justified.

Root independently derived the spherical calculation and ran exactly
one tiny, standard-library rational-arithmetic certificate. This was
the only numerical execution of this turn; it used no network, target,
random samples or training arrays:

- `certify_spherical_stein_constants_v1.py`, SHA256
  `53828c296325679ed4151234aa159551e1c20e69651d89abc750b12f5945f224`.
- `spherical_stein_constants_certificate_v1.json`, SHA256
  `8b0e783690c9780c605e1fd6ac9b96b3d8b44980b3bf470fdbcc14eaddbe8451`.

Certificate PASS: exact quadratic-harmonic check at dimensions
3, 4, 8, 16, 64 and 1024; n=1024 Gaussian cost infimum
523775/270336 = 1.937496300899621. Agent independently read the code and
receipt and found no issue. The code does not prove the general Bochner
identity or measure the network's nonlinear angular energy. The note
uses n(2n-1) per dense matrix-vector product; the absolute budget
yardstick in the certificate uses leading 2n^2. Their relative
77/16 and (46+31q)/16 counts agree. Neither is a native FNP meter.

## Carried-forward closures and authority

The completed learned response-conditioned pair-memory pilot remains
CLOSED: next-mean proxy ratio to zero 0.9911990675, history/RESET
1.0002367769, only 1/4 held-out history wins. Its source-only optimistic
278.775B arithmetic envelope is not an achieved adjusted score; the
information prerequisite failed. No restart, larger model, extra
epochs, gates, gains or rollout is authorized under that fixed arm.

The earlier asynchronous request for exactly one score-aware
re-evaluation of the already-trained response-row reduction has NO
reply. It remains closed under its original raw-error gate. Do not
port, meter, panel, refit or re-evaluate it on generic submission
authority. No cold-trace exception was requested. The existing
negative-family receipts remain binding for their specified arms;
do not reinterpret today's scoped proofs as universal impossibility.

## Processes, resources and continuation

At 17:46 PDT, process inventory found no Python, whest, training or
card job. Only the pre-existing `winner_lock.sh` helper remained.
Both agents completed their source/proof tasks. No numerical session
needs polling, restarting or resuming. Swap is unchanged at 816.75 MiB
used of 2048 MiB allocated. Preserve <=6 nice-10 CPU card workers and
one MPS job at a time; no threadpoolctl import in AP, MPS fallback off.

No forum post, external message, submission, commit, promotion,
deletion, frozen-file edit or new infrastructure installation occurred.
Goal stays active: these are useful negative results, not completion
and not a blocker caused merely by difficult research. A next numerical
experiment requires a materially different information/cost mechanism,
an explicit source audit and a bounded falsifiable protocol. There is
still no demonstrated route to the remaining raw-error reduction.
