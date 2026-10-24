# CS2 one-attempt exploratory Phase-2 board-promotion scaffold

Status at seal: **HOLD — target-free scaffold only; the fixed full-width
comparator and literal-meter receipts do not yet exist or bind this promotion.**

## Question

If and only if the exact frozen CS2 source passes its separately preregistered
three-seed width-1024 comparator, this board-specific change-size gate, and its
literal Phase-2 meter, does one exact single-file archive complete all 100
Phase-2 MLPs with zero failures and return a finite unrounded aggregate raw
final-layer MSE strictly below `1e-8`?

This preregisters one candidate and at most one remote-create attempt. It does
not authorize full-width execution, metering, packaging, network or credential
access, upload, submission, or a graph edit.

## Exact frozen candidate

Candidate count is exactly one. The candidate is
`../fullbudget_k21_k4_closure_q24_connected_shutdown_v2/estimator.py`, 72,277
bytes, SHA-256
`4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779`.
No source mutation, candidate substitution, threshold tuning, row-specific
branch, fallback, or retry is allowed.

The source HOLD is
`../fullbudget_k21_k4_closure_q24_connected_shutdown_v2/SOURCE_HOLD_V2.json`,
SHA-256
`c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c`.
The additive independent strict audit is
`../fullbudget_k21_k4_closure_q24_connected_shutdown_v2/INDEPENDENT_STRICT_AUDIT_CS2.json`,
SHA-256
`cba8bfbb437f32aa377ca72da74b9504894e6380429e2acf0e5a059d7ef8afcb`,
with exact status
`PASS_INDEPENDENT_EXACT_BYTE_SOURCE_AND_TARGET_FREE_CARRIER_AUDIT__IMMUTABLE_HOLD__FULLWIDTH_UNMEASURED`.
The historical `independent_source_audit: false` field inside the earlier
source HOLD is preserved rather than rewritten; this later exact-byte audit is
the authoritative additive closure of that gate.

## Pre-board target-free gates

The fixed comparator is the one-use, weights-only panel already preregistered
for official-width seeds `20260824`, `20260825`, and `20260826`, in that order.
Its own PASS remains necessary but is not sufficient for this promotion.

Before any future comparator result is opened, this scaffold additionally
freezes the following board-specific promotion gates:

1. For each of the three fixed seeds independently, the final-layer
   parent-versus-CS2 disagreement MSE must be finite and `<= 1e-8`.
2. There must be zero parent-finite-to-CS2-nonfinite or raised regressions.
3. The exact aggregate count of pairing-alpha, connected-star, CP-carrier, and
   combined-carrier invariant violations must be zero, and every individual
   constituent violation count must also be zero.
4. Every original comparator source-stability gate must pass, including at
   least one activation over the panel, the no-activation byte-parity rule, and
   both parent-scale guards.

The `1e-8` disagreement threshold is the user's target scale, not a value
selected from future comparator output. It is an additive board-promotion gate
and does not retroactively alter the comparator's registered PASS status.
Equality at `1e-8` passes this change-size gate.

Only after those gates pass may the separately preregistered one-use literal
meter be released. Its exact legal gate is: billed FLOPs at most
`2^41 = 2,199,023,255,552`; wall time below 120 seconds; residual time at most
0.4 seconds; descendant RSS strictly below 8 GiB; zero failed MLPs; and one
finite float32 output of shape `[16,1024]`. Comparator telemetry is not a
substitute for this literal meter.

## Post-hoc operational motivation only

The previous frozen Q24 archive's submission `328210` completed 100/100 MLPs
with zero failures and returned exact raw MSE
`5.765125013112993e-7` and adjusted score `5.587487766740284e-7`. Its 50 rounded
public rows were concentrated in a small tail, but that observation is used
only as post-hoc operational motivation to evaluate an already target-free
certificate-triggered mechanism. It does not identify a cause, choose the CS2
trigger, tune a threshold, weight a row, or provide training data. The result
and independent audit are bound by SHA-256
`d8b488eddee2cd0c70aaaa75d53d1b519a75ccdbfdfbda7e78fb7896b7c48d1b`
and `3a50f5cee86af8fff225ec5bda965cc6f35abd98a8278e7b37e111cb570b3f5e`.

## Package and local smoke boundary

Packaging remains forbidden while either target-free receipt binding is null.
After both pass and a root-controlled release is sealed, exactly one archive
may be created. It must contain exactly two regular top-level members and no
others:

- `estimator.py`, byte-identical to the 72,277-byte frozen CS2 source; and
- `manifest.json`, using schema `1.0`, API `2.0`, entrypoint
  `estimator.Estimator`, whestbench `0.16.0`, flopscope `0.12.0`, numpy `2.4.6`,
  and Python minimum `3.11`.

Before a remote create request, the pinned client must pass package validation
and an independently recorded source/package smoke with finite output shape
`[2,4]`. Archive member names, types, byte counts, hashes, embedded manifest,
and exact estimator parity must be rebound in an additive final-ready receipt.
No archive exists or is authorized by this scaffold.

## Board metric and one-attempt law

The primary board gate is all of:

1. exactly 100 of 100 configured MLPs complete;
2. zero MLPs fail;
3. a finite unrounded aggregate raw final-layer MSE is returned; and
4. that raw MSE is strictly `< 1e-8`.

Equality at the board threshold, a larger value, missing/nonfinite aggregate,
partial completion, any failed row, or grader failure is NONPASS. Adjusted
score, all-layer MSE, partial-row arithmetic, utilization, or a fallback value
cannot rescue the primary gate.

One platform-visible submission or remote create request consumes the one
attempt. A watch timeout after a submission ID also consumes it. An ambiguous
post-create transport failure is classified as a possible consumed attempt and
forbids retry until the platform record is reconciled. A local preflight
failure before any remote create consumes no attempt. There is no same-archive
retry, regrade farming, fallback, threshold change, candidate substitution, or
target-driven mutation.

## Terminal result receipt and independent audit

After the one attempt, the root must create one immutable
`BOARD_ATTEMPT_<id>_RESULT.json` from read-only authenticated terminal status
and the public evaluation page. It must bind the exact archive and submission
provenance and preserve:

- terminal grading state, timestamps, smoke result, and attempt consumption;
- configured/completed/failed/in-flight counts and the complete failure
  breakdown;
- exact adjusted and raw aggregate scores, plus public display values;
- arithmetic over every public per-MLP score row, with rounding bounds;
- public outlier indices, values, and contribution shares, explicitly scoped
  to rounded public rows;
- min/mean/max FLOPs, predict wall, backend wall, and residual telemetry over
  all returned rows; and
- a direct replay of the 100/100, zero-failure, finite-raw, strict-`1e-8`
  verdict.

A separate agent must then add
`INDEPENDENT_BOARD_RESULT_AUDIT_<id>.json`, binding the terminal receipt hash
and independently replaying candidate/archive provenance, score arithmetic,
row counts, tail arithmetic, telemetry limits, and the terminal verdict. No
targets, private evaluation state, credentials, or unobserved causal claim may
enter either receipt.

## Fail-closed activation

`PROMOTION_MANIFEST.json` holds the comparator and literal-meter receipt hashes
and measured values as explicit null blockers. Only the root may populate
those fields after reading immutable PASS receipts and may then bind an exact
package, smoke audit, final argv, and additive final-ready audit. The frozen
scientific law in this file is not mutable.

The finalizer must run `verify_promotion_hold.py --require-ready` successfully
before any upload. Until then, the command template's placeholders are literal
and non-executable.

## Forbidden while HOLD

- full-width or meter execution from this namespace;
- target, dataset, score, private-board, network, or credential access;
- source mutation, packaging, upload, submission, or retry;
- filling a hash from mutable or unverified bytes;
- tuning, fallback, candidate substitution, or graph mutation.
