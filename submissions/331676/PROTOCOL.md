# Final readout adaptation to full lower moments

The previous goal turn completed an explicitly authorized exact upload and
verified a live grading wait. Submission331672 remains enqueued at the first
poll this turn. Its source and archive remain unchanged. The goal is still
approximately1e-9 adjusted, with intermediate3e-9 and2e-9 banks outstanding.

The full lower-moment source has improved local accuracy on14+8 reused
networks, while retaining the final linear head inherited from the older
physical trajectory. This study tests whether a new final65-coefficient
readout improves that changed trajectory. It does not rescue any covariance
gradient proposal, change the underlying physical estimator, or claim a new
feature family. Older native-head, latent-covariance and frozen-observer fits
remain separate with their recorded results. This is an incremental accuracy
step; passing it would not complete the overall goal.

Freeze the unchanged submitted source bbae60ddfad2718da44d4bcc6cf50110998675fcc5f7a79e6bc4dd7c5b4b9d12.
Capture the already-computed final H15, sigma, scalar host, original delta,
head/bias and fixed scale parameters through passive wrappers. The wrappers
call each original function exactly once, do no added FNP array arithmetic
inside its budget, and never alter an operand. Qualify on35/100/35 against
the saved identical-source triple: all16 output bytes, complete bills and
weight hashes must match;35 features repeat exactly; all caches retire.

Fixed TRAIN20=[35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58].
Reuse qualified cold35; acquire each remaining network cold, serially. Seal
all captures, measured bills and source/weight/target bindings before decoding
existing ordinary1B mc_post reference means. No new samples or dataset copies.
These are reused public research networks, not globally untouched holdouts.

Use five fixed network folds by roster index modulo5. Fit16 networks and
predict the other4. Only the64 final head coefficients and final bias vary.
All earlier heads, neural encoders, NODE coefficients, physical operations,
selectors and schedules remain frozen. Define the65 basis columns as
scale*[H15,1], where scale is the original FP32 sigma*rms*(1+.25*tanh(gain))
interpreted in FP64 for regression. Fit a correction about the old head to
truth minus the saved UNCLIPPED parent output. Normalize each column by its
fit-only weighted RMS; weight each row by its network's measured adjusted
score multiplier and normalize weights to sum1. Solve one ridge system with
fixed lambda1e-3, penalizing every standardized correction coefficient.
No lambda, feature, fold, seed or clipping selection follows the result.

Predictions use the unchanged scalar host and FP32 readout arithmetic with
the fitted head rounded to FP32, followed by the existing nonnegative clip.
Report all network/fold raw and adjusted scores and normal-equation residuals.
The development decision is pooled adjusted ratio<1, at least3/5 fold wins,
and no fold ratio>1.05. These gates ask whether the actual objective improves;
there is no claim that a small improvement alone reaches the requested target.
Check the ridge solver on an independent synthetic regularized system before
reading targets. Reproduce the saved original readout to<=2e-7 maximum error.

Only if cross-validation passes, fit once on all20 with the same rule and
freeze an edited estimator changing only those65 literals. Then evaluate the
fixed disjoint confirmation8=[37,39,42,48,59,63,66,67], requiring pooled
adjusted improvement, >=5/8 wins and maximum individual ratio<=1.10. This is
a new check for this fit, although all networks have been used in research.
Native same-history outputs, first15 parent bytes, bills, exact own repeats,
client/runtime, contract and package qualification are required before any
upload. Existing standing authorization covers a qualifying upload.

One nice10 worker, BLAS1,8GB group-RSS guard,256MiB emergency swap-growth
cutoff; completed stages require zero added swap. Full calls have100s limits;
qualification triple240s; small fit60s. Anticipated artifact size<20MB. No GPU,
concurrent numerical work, new samples, broad hyperparameter search or edits
to prior records. Store adverse results and keep the goal active.
