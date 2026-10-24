# Native final-error graph head: fixed pilot closed

Recorded 2026-09-26 21:39 PDT. **No new banked improvement.** Official S31 submission
332517 remains **2.1736465056908115e-9 adjusted**, with 100/100 evaluations
complete and zero failures in its preserved official receipt. Its source,
submission archive and retained thin-star source hashes are unchanged.

The graph head improves frozen FIT128 raw MSE by 4.84726%, but worsens
excluded CHECK32 raw MSE by **2.97073%** (paired network-bootstrap 95%
interval **[+2.31249%, +3.67570%]**, **2/32 wins**). The local-feature
control improves FIT by 0.50001% and is **0.07575% worse** on CHECK32
(95% interval [-0.16869%, +0.33286%], 16/32 wins). Graph versus control
is +2.89280% raw MSE. The FIT gain does not generalize in this fixed test.

All original graph information gates and all prospective smaller-gain
paid-screen gates fail. `survivor.json` selects neither head. Both heads
are closed with no CHECK-guided sign, amplitude, epoch, feature or
architecture rescue. No changed-head native candidate, metered inference
qualification, VALID40 evaluation, package or submission follows.

## Scope and controls

The fixed split contains 128 FIT and 32 excluded CHECK networks, all from
the existing TRAIN160 research set. These are not pristine independent
holdouts: the underlying estimator and earlier research have used TRAIN.
VALID40 was unopened for this pilot. The split and all model choices were
fixed before this pilot read labels.

Passive native S31 capture supplies final scalar/state features plus 57
transported fields from layers 12, 13 and 14. Those fields use signed and
absolute correlation-neighbor averages, soft neighbors and a covariance
power-subspace projection. Both heads have a fixed 151-512-256-1 SiLU
architecture, zero initial output, identical data order/seed, and 40 epochs
(1,280 updates). The control zeros the 57 graph columns. Normalization uses
FIT only. Terminal predictions are rounded to float32 before scoring.
Both fitted weights and CHECK predictions were sealed before assessment
read CHECK labels; no approximate native trajectory derivative is used.

This is an offline final-error readout on native captures, not a deployed
or cost-qualified changed native head. It closes this feature/architecture/
training instance, not every possible graph estimator.

Tiny target-free permutation, neighbor and transport checks pass. All 160
captures finish without reading targets. Twenty independent historical
all-16-layer prediction files match byte for byte. The original matched
35/100 smoke also matches complete bills; the other historical comparisons
do not certify bills because process orders differ. Capture bytes and both
fitted artifacts match their existing seals at closeout.

## Prospective retention and provenance

`control_retention_addendum_v1.json` and the superseding retention-only
`banking_retention_addendum_v2.json` were written before either fit began
or any new FIT/CHECK score existed. The second permits an already fixed
head with at least 3% raw gain, a 95% upper ratio below 0.99 and at least
24/32 wins to enter paid native screening. It preserves the original 5%
graph information gate and requires at least 2% actual adjusted gain plus
further qualification before release. Neither arm passes either rule.
No model, prediction, primary assessment or threshold was changed after
the CHECK result. The pre-fit source seal remains exact.

`assessment.json`, `survivor.json`, `precheck_seal.json`, and both
`fit_*/result.json` files are the scientific receipts. `native_helper_draft.py`
and `build_native.py` are unexecuted drafts; they establish no inference
parity, bill or performance result. Historical launch/in-progress/session
records are superseded by `closeout.json` and checkpoint 24.

## Resources and continuation

All 21 guarded groups completed successfully and retired. Peak sampled
RSS was 6,238,453,760 bytes (6.238 GB); added swap was zero.
The serial capture contains 121,402,240 bytes, with no full covariance
dump. About 436.9 GiB is free at closeout. The process census
finds no remaining capture, fit, assessment or supervisor workers.

Board submissions remain authorized. No permission, upload or grading is
pending. Thin-star's earlier 0.57346% TRAIN8 adjusted gain remains below
the submission bar and lacks held-out/actual-worker qualification. No new
candidate is selected. The 2e-9 and approximately 1e-9 goals remain unmet;
the goal is active. Restart from `../continuation_checkpoint_v24.json`.
