# Frozen cap4096: measured objective failure

The previously unscored terminal native candidate now has an actual error
measurement. It fails the fixed objective gate decisively, even using its
favorable saved sequential bills against Lazy's independent-cold bills.

| Evaluation | Reference adjusted | Terminal cap adjusted | Ratio | Cap wins |
| --- | ---: | ---: | ---: | ---: |
| Original 40, uncapped old bank | 4.766355577729645e-9 | 7.711108463947432e-9 | 1.617820646864215 | 0/40 |
| Original 40, untrained cap | 8.26458112816147e-9 | 7.711108463947432e-9 | .9330307663956391 | 37/40 |
| Common 8, current Lazy | 3.8588427268781516e-9 | 8.052305190410887e-9 | 2.0867150491321773 | 0/8 |

Training improved the cap's own adjusted error by 6.70%, but the resulting
model remains substantially worse than both the old uncapped bank and Lazy.
Original-40 terminal raw MSE is 5.47484085348415e-8 at mean measured sequence
cost 309716647338.85 FLOPs. All scores are means of per-network raw MSE times
max(.1, that call's own FLOPs/2^41), not products of unrelated means.

The original acquisitions initialize once and preserve candidate state.
These are measured sequence scores, not fabricated cold bills. The common-eight
comparison favors the cap by using those same sequence costs against Lazy's
measured independent-cold costs. It still loses on every network. No new cold
run, runtime qualification, archive, submission or fit is warranted for this
fixed candidate. These are reused ordinary1B research references; no claim
of pristine holdout or noise-corrected panel equivalence is made.

All original native acquisition, source/weight hashes, five-artifact input/bill/
lifecycle records and exact repeat checks pass. Predictions and bills were
sealed before reference access. The old native/MPS <=1e-5 fidelity gate still
fails on 7/40 networks, maximum 3.4943222999572754e-5. Actual shadow/native final
MSE ratios range .9984798265621299 to 1.002038062621613. Thus numerical drift is
small relative to the measured objective deficit; it is not the missing gain.
No gradient fidelity, coadaptation success or native training claim is inferred.

This audit performed zero network predictions and zero training updates.
The serial guarded scorer completed in 3.7016 seconds, with peak RSS
69779456 bytes, zero added swap and the process group retired. Existing model,
training, failed judge and later retirement diagnosis remain unchanged.

The strongest qualified local model remains Lazy at noise-corrected14
3.816239868909105e-9. Its explicitly approved exact archive was accepted as
AIcrowd #331599; the latest poll is still submitted/queued with no score.
Official best remains #331584 at 4.015535762022727e-9. Approximately1e-9 remains
active and unmet. See source_review.json for recovered later source tests.
