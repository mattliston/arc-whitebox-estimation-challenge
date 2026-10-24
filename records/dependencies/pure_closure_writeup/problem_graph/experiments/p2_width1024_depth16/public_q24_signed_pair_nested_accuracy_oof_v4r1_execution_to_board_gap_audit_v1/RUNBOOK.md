# OOF V4R1 execution-to-board runbook

This is a control-only map. It opens no bank, staging, row, weight, target, or
score payload and authorizes no numerical run. CPU is the sole authoritative
device; MPS is not authorized. Resource observations cannot veto the frozen
accuracy experiment, although deployment must later satisfy the board meter.

## The shortest live path to one raw-MSE result

1. Complete Runtime V5 generation once. Do not retry. The transaction must
   atomically publish `THREE_BANK_BUNDLE` with exactly the six bank files and
   `THREE_BANK_GENERATION_RECEIPT.json`; staging and failure names must be
   absent.
2. Execute the six direct argv entries in
   `public_q24_signed_pair_lean_v5_post_generation_handoff_v1r1/COMMAND_PLAN.json`
   in order. Step 5 must be performed by an agent different from the seal
   issuer. This produces, in order:
   `FULL_BUNDLE_HANDOFF_VERIFICATION_V1.json`, `POST_GENERATION_SEAL_V5.json`,
   `INDEPENDENT_REVIEW_REQUEST_V1.json`,
   `INDEPENDENT_POST_GENERATION_REVIEW_V5.json`, and
   `OOF_CONSUMER_HANDOFF_V1.json`.
3. Use only
   `public_q24_signed_pair_nested_accuracy_oof_v4r1_execution_authority_builder_v1r1`.
   Its package root is
   `6e80d902b17e396b38a85527584084f725f8c2f3bc7854066cb9135566ee2b9d`,
   helper root is
   `572477ccfec5768aea3916db01a6c2b19860efd0b8178b0f2777c4817d176c9b`,
   and its
   independent static GO is
   `public_q24_signed_pair_nested_accuracy_oof_v4r1_execution_authority_builder_review_v1r1/INDEPENDENT_STATIC_REVIEW_V1R1.json`,
   SHA-256
   `8c3fee650c595ada2b0bb3208c41ce84b7648e532c2f4d301e6b3479a16bd3fd`,
   schema
   `public-q24-nested-oof-v4r1-execution-authority-builder-independent-static-review-v1r1`,
   status
   `GO_FROZEN_FAIL_CLOSED_POSTSEAL_V1R1_AUTHORITY_PATH__NO_LIVE_AUTHORITY`.
   Treat that review as a required external sequencing precondition before the
   first `issue-anchor` call. The reviewed V1 builder is unusable because it
   pins the held V1 postseal helper.
4. Create a canonical anchor request at a separate path, with exactly
   `schema`, `authority_id`, `issued_utc`, and `confirmation`. Its confirmation
   is
   `ISSUE_FROZEN_NESTED_OOF_V4R1_ANCHOR_AFTER_V5_BANK_REVIEW`; its timestamp
   must be no more than five minutes old.
5. Run the successor helper as direct argv with
   `/Users/matt/anaconda3/bin/python3`, first `preflight-anchor --request
   <absolute path>` and then `issue-anchor --request <absolute path>`. Set
   `PYTHONDONTWRITEBYTECODE=1` in the environment. The live authority directory
   must be absent before issuance.
6. A different agent reviews the exact anchor and writes only
   `public_q24_signed_pair_nested_accuracy_oof_v4r1_execution_anchor_review_v1/INDEPENDENT_EXECUTION_ANCHOR_REVIEW_V4R1.json`.
   It must use schema
   `public-q24-signed-pair-nested-accuracy-execution-anchor-independent-review-v3`,
   status `GO_EXACT_ONE_SHOT_ACCURACY_ANCHOR`, issuer
   `independent_reviewer`, and the exact anchor/package hashes. Capability and
   release must not yet exist.
7. Create a fresh canonical capability request with exactly `schema`,
   `capability_id`, `issued_utc`, and `confirmation`. Its confirmation is
   `ISSUE_ONE_NESTED_OOF_V4R1_TARGET_CAPABILITY_AFTER_ANCHOR_GO`; run the same
   helper by direct argv with `issue-capability --request <absolute path>`.
   This creates exactly one observer-fit target capability, no retry.
8. Create a fresh canonical release request with exactly `schema`,
   `release_id`, `issued_utc`, and `confirmation`. Its confirmation is
   `RELEASE_EXACTLY_ONE_CPU_NESTED_OOF_V4R1_RUN_NO_RETRY`; run the same helper
   by direct argv with `issue-release --request <absolute path>` and then
   `final-gate`.
9. Execute the stored `argv` from
   `public_q24_signed_pair_nested_accuracy_oof_v4r1_execution_authority_instance_v1/EXACTLY_ONE_CPU_NESTED_OOF_V4R1_COMMAND_V1.json`
   directly, without shell reinterpretation. The frozen argv shape is:

   ```text
   /Users/matt/anaconda3/bin/python3
   <P2>/public_q24_signed_pair_nested_accuracy_oof_v4r1/nested_accuracy_runner.py
   --execute
   --release
   <P2>/SIGNED_PAIR_NESTED_ACCURACY_EXECUTION_RELEASE_V4R1.json
   --capability
   <P2>/public_q24_signed_pair_nested_accuracy_oof_v4r1_execution_authority_instance_v1/SINGLE_USE_TARGET_CAPABILITY_V4R1.json
   ```

The runner claims before any dynamic source, bank, row, weight, or target byte;
consumes the capability by same-inode link/unlink; performs one post-claim full
audit; then runs the frozen nested science. Any post-claim failure is terminal
and no retry is permitted.

## ETA from the reviewed consumer handoff

- Live anchor, independent anchor review, capability, release, and final gate:
  about 10–30 minutes if the reviewer is ready.
- Runner's full bank preflight plus CPU nested OOF: about 3–5 hours on the
  currently measured path.
- Therefore the first raw-MSE result is approximately 3.2–5.5 hours after the
  reviewed consumer handoff, exclusive of machine interruption.

## Why a PASS is not yet a board package

The result schema is
`public-q24-signed-pair-nested-accuracy-oof-result-v3`. A pass explicitly says
`PASS_SIGNED_PAIR_RAW_HURDLE__NO_SUBMISSION_AUTHORITY` and opens an
integration/resource-optimization child only. No repository consumer currently
seals this output or builds a signed-pair deployment estimator.

The deployment choice is now frozen in advance: after an independently sealed
OOF PASS, select one `H` from the four fold-pure inner-selection receipts and
fit one fresh model on all 688 observer-fit rows. For each `H`, concatenate the
`metrics[H].per_network[].pair_on_mse` vectors from all four
`SELECTION_OUTER_FOLD_k.json` files (4 x 516 observations, with every network
appearing exactly three times), take the float64 mean, retain values within the
frozen 0.0025 relative tolerance of the minimum, and choose the smallest
retained `H`. This selection consumes only the completed OOF receipts; it does
not reopen the target bundle.

This is the usual nested-CV boundary: the OOF score estimates the complete
selection-and-fit procedure, not the exact bytes of the eventual full-data
model. An equal-weight average of the four outer models is diagnostic HOLD
only, because three of those four models trained on each observer-fit row. A
post-refit observer-fit score would likewise be training error, not new
held-out evidence, so the shortest honest path does not run one.

The next problem leaves are:

1. Seal and independently review the exact completed OOF output.
2. Apply the frozen pooled fold-pure `H` law and write an immutable selection
   receipt bound to the reviewed OOF output.
3. Issue a refit-specific anchor, independent review, new one-use observer-fit
   target capability, and final release on an acyclic DAG. The capability must
   bind the anchor, never the later release hash. The OOF capability will
   already be consumed and is not reusable.
4. Export the chosen state into the CPU/flopscope estimator and prove exact
   parity with the authoritative torch computation and target-blind online
   action construction.
5. Run the literal production meter and optimize until finite float32
   `[16,1024]`, FLOPs `<=2^41`, wall `<120s`, residual `<=0.4s`, and descendant
   RSS `<8GiB` all pass. This is downstream of accuracy; a resource miss does
   not erase the OOF result.
6. Freeze and independently audit `estimator.py`; package exactly
   `estimator.py` and `manifest.json`; run pinned package validation; issue a
   BANK_READY receipt.
7. Bind fresh root submission authority to that exact archive. Existing score
   claims and submission/fire receipts are candidate-specific and cannot be
   reused.

At the frozen raw hurdle, adjusted score below `7.7e-9` requires a deployed
multiplier below `0.6196849852090025`, equivalent to about
`1.362701693591e12` FLOPs only in the constant-FLOP case. The authoritative law
is `mean_i[MSE_i * max(0.1, FLOPs_i/2^41)]`; if FLOPs vary by network, the
held-out per-network errors must be paired with their own literal bills and
averaged. `mean(MSE) * mean(factor)` is not a valid shortcut. This deployment
constraint should guide optimization after accuracy is known; it must not be
used to suppress an accurate but initially expensive result.

## Fastest post-PASS path

Once the OOF result is durable, there should be no second 688-network target
score. The direct critical path is:

1. Seal and independently review the exact OOF output inventory and PASS.
2. Materialize the pooled `H` receipt using the preregistered rule above.
3. Issue a fresh all-688-refit anchor, have a different agent review it, then
   issue one target capability and one final CPU release.
4. Run one final J fit and one selected-H signed-pair fit, save one deployable
   state, and consume the refit capability before decoding any target value.
5. Seal/review the refit state; export it target-blind; prove authoritative
   torch-to-export and online-action parity.
6. Run only the fixed production meter fixtures, then freeze, audit, validate,
   and archive exactly regular top-level `estimator.py` and `manifest.json`.
7. After BANK_READY, obtain fresh root authority for exactly that archive and
   execute the direct `whest submit ... --watch --watch-timeout 3600 --format
   json` argv once.

The final refit is much smaller than nested OOF: OOF trains four outer fits plus
the full all-H inner-selection grid, whereas the final stage trains only the
selected spelling once. A reasonable control-plane ETA after an OOF PASS is
roughly 2--4 hours to a reviewed package if the export/parity/meter gates pass
without repair; submission timing remains outside this static audit and needs
fresh exact-archive authority.
