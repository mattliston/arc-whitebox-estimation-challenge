# One authorized response-row adjusted-score re-evaluation

Preregistered September 6, 2026 PDT, after the user's explicit narrow grant.
This document supersedes only the pending-permission status and raw-only
stopping gate for this one fixed arm. It does not revise or erase
RESPONSE_ROWS_MATCHED_ROUND_V1.md or its CLOSED receipt. No other failed
family is reopened. No training, refitting, threshold, capacity, or head
selection is authorized by this exception.

## Goal and confirmed baseline

Phase 2 only. Official bank #329926: adjusted 4.982231460183304e-9,
raw 2.2448352261505987e-8. Final goal strictly below 1e-9; validated
submission milestones below 4e-9, 3e-9, and 2e-9. Small qualified gains
are retained as possible components, not described as milestone banks.
No forum posts, commits, promotions, or frozen-file changes.

## Two routes and selection, before measurement

1. **Selected: frozen response-row reduction in the current runtime.**
   Historical learned full-final raw changes versus its then-matched parent
   were +3.1204% on rank35 and -0.3523% on rank100, using stored 1e9 truths.
   A separate unlearned, D5 scalar-final comparison saved 5.85% compute;
   that number is not a learned/current-bank cost estimate or score.
   Source census suggests a possible low-single-digit adjusted improvement.
   Complete candidate planning range is 0.47--0.52T FLOPs per MLP (later
   source audit narrows the plausible middle to 0.47--0.50T), versus about
   0.49T for the bank. This is not a bound: final geometry, restored final
   plane, covariance/star work, all full heads, and FP64 radial restoration
   must be metered together. The unresolved assumption is that savings
   survive restoring the response model's own full final head. Cheapest
   decisive test: fixed source/head parity and actual output plus complete
   native bill on ranks35/100, against the untouched current bank in the
   same evaluator; repeat35 checks persistence. No archived raw/new-cost
   multiplication is permitted.

2. **Raw-accuracy route: independent full-matrix fourth-cumulant state.**
   Corrected privileged SSC fits explain roughly 0.61--0.69 of held-out
   marginal fourth-cumulant variation, and a full-trace oracle removes
   about 39% of one isolated fourth-driven next-mean residual. Neither is
   final-layer raw improvement. No credible final adjusted magnitude is
   presently established. Congruence transport alone costs about 62.3B
   FLOPs per retained mode over depth with a diagonal final read (64.4B
   with a full final matrix). Mode birth/compression and the necessary
   all-distinct fourth-to-third feedback are not yet priced or specified;
   therefore complete cost is unknown. Cheapest decisive next step is a
   source-level causal recurrence with a complete contraction/billing
   inventory, not another rank sweep or numerical experiment. No run of
   this route is justified yet. Compute-only savings cannot meet 1e-9:
   even the multiplier floor needs raw below 1e-8.

## Immutable configuration and provenance

Base: candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py
SHA256 3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757.
Frozen response payload: payload_response_rows_v1.npz
SHA256 6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c.
Response map: radial_response_rows_v1.py
SHA256 bb496d4a205d4669d35050e1f56560fbd303a8125a215cb560a64ee4e2a785f5.
Archived closed-loop NPZ references:

- rank35: 8c428c6a7cc8ede17a6b10464719d35513413aec17f086f02809a9a5253f69ca.
- rank100: c1e2d0dca91f57e52f92f334404442aad153097ec453b536741280631c3bd194.

The derived candidate retains D4, tile8192, bulk-copy runtime and the
fixed .01 response selection. It preserves every CP column, all unit
diagonals, active covariance/star coordinates, and exact omitted radial
source restoration. Use the frozen response P for every full-feature
head, including layer15. Its centers/scales, state, encoder, messages,
and heads are a single coadapted payload. Do not graft the bank's Q
scalar-final head onto it. Remove the final scalar shortcut; retain the
full-path final hook then break, without outgoing layer15 CP birth.
This compares the faithful old method in the current runtime against
the actual bank; it does not claim response P and bank Q compatibility.
All estimator numerical operations remain FlopScope operations.

## Fixed tests and score-aware decisions

1. Review exact-count/AST source changes; hash the derived source and
   builder before execution. Check payload value identity and all-layer
   head routing. The no-fit constraint includes the final head.
2. Native FlopScope0.12/Whest0.16, one BLAS thread, separate process per
   candidate, fixed rank35/100/35 sequence, budget 2^41 and 120s per call.
   Preserve served float32 outputs and full per-call bill before scoring.
   Capture response pre-head features/deltas on the first two calls.
   Existing parity gate remains output max error <1e-5, delta max error
   <1e-6 versus the fixed archived full-loop reference, finite (16,1024)
   float32 output, byte-identical repeated output, peak RSS <8GiB.
   Report feature discrepancies and every layer, without post-result
   tolerance changes. Diagnostic copies are not deployment residual-time
   evidence. Any failed compatibility gate requires a documented source
   bug diagnosis; no scientific/configuration change is allowed.
3. Score the saved actual outputs of each configuration against the same
   stored 1e9 targets on35/100. For each net compute raw final MSE times
   max(0.1, complete measured FLOPs / 2^41), then average those products.
   Report raw, bill, multiplier, adjusted, source, net and reference.
   Include first-call initialization equally and report warm repeat cost.
   A lower two-net mean adjusted score advances to validation regardless
   of raw-only worsening. An unchanged/worse mean closes this exception;
   no threshold or head search follows. These two nets are development
   evidence, not the independent panel or official board.
4. If advanced, use the existing independent 8-public +6-virgin panel
   and its stored independent 4M truths for candidate and current bank.
   Require lower mean adjusted score on BOTH groups to retain a validated
   gain. Compute per-net products, not the product of group means. Keep
   noisy-panel evidence separate from the 1e9 development truth results.
5. Before any release, run the exact unwrapped source through persistent
   NumPy-free client/server0.12 contexts35/100/500/3016500/35. Require
   byte repeat, all finite correct-shaped outputs, full per-call budget,
   local one-thread wall <60s, residual <0.4s, and combined sampled RSS
   <8e9 bytes. The official 120s wall ceiling is not a local safety target.
   Local macOS checks are not a guarantee of the Linux grader. Require
   Whest0.16 validation, matching source/archive hashes and all existing
   numerical checks. Submit once only when milestone/release policy is
   met; a positive small component alone is not a new milestone.

## Resources and preservation

At most six nice-10 CPU card workers, one MPS job; this fixed evaluation
needs no MPS/training. Use one candidate process at a time for peak-memory
measurement. Check swap and live jobs before heavy work. New outputs must
be versioned/exclusive-create. Preserve every old source, payload, label,
output, raw-only closeout, and official bank receipt unchanged.
