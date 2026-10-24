# Fixed terminal nonlinear ensemble — negative information result

The five-fold, four-member bootstrap ensemble fails its predeclared gate.
Adjusted ordinary1B-target development errors, including the declared readout
work reserves, are:

| Arm | Pooled adjusted error |
|---|---:|
| Unchanged Read5 |3.989137621921596e-09|
| Same TRAIN-only affine control |3.974845830335309e-09|
| Nonlinear ensemble |4.134239881657254e-09|
| Mean of individual-member scores |4.246380487610553e-09|

Ensemble/unchanged=1.036374342900148; ensemble/control=
1.040100687706043. **0/5 folds win** against both baselines.
The required ratios were<=.95 and<=.97, with4/5 wins. No CHECK access, full-data
fit, native port, added epochs or width/seed/penalty/member-count sweep follows.
This closes this fixed4,097-parameter terminal nonlinear bagging test, not all
possible learned estimators or every information source.

Averaging helps relative to the individual fits, but both are worse than the
unchanged estimator. All20 training objectives decrease: final normalized
bootstrap losses range0.879335391 to0.923288178, while every outer
fold's ensemble score worsens. This is measured failure to generalize across
these held-out networks, not an optimization failure or a reason to train longer.
The mean-prediction MSE/variance identity is verified separately on unclipped
FP64 predictions (maximum error5.56e-22).
No identity is asserted for individually clipped/rounded predictions.

## What was actually fit

The existing Read5 terminal126-input captures and sigma*[1,H15] control are
reused without another physical prediction. Five outer folds are ordinal%5
on the fixed20 networks. Every normalizer, residual scale, affine coefficient
and nonlinear projection comes only from that fold's16 TRAIN networks.
Each of4 members has126->32->1 SiLU layers and its own presealed initialization
and16-network bootstrap draw. The four-member mean is the only proposed arm;
individual members cannot be selected.

At each of200 full-batch updates, its correction is projected off the common
TRAIN affine span before the loss. The same TRAIN projection is applied to
held-out predictions. Model arithmetic is FP32, projection/loss reduction FP64,
with the actual final nonnegative FP32 serve. AdamW/cosine/gradient cap and all
200 updates were fixed in advance; exactly4000 updates/20 terminal payloads
completed. No backbone derivative, recurrent fit or physical parameter changed.
The first zero-head step correctly has zero trunk gradients; the nonzero-head
algebra fixture independently exercises those paths.

These are repeatedly studied TRAIN networks, some used by the base model.
The test is exploratory network cross-validation, NOT independent validation.
All20 initial arrays and bootstraps were sealed before this trainer opened
original1B target files. Capture/target hashes and canonical weights match
the original target manifest. Every held-out prediction and terminal payload
was sealed before aggregate scores were computed. No new Monte Carlo data,
held-out CHECK4 truth or broader14 panel was opened.

## Qualification and cost limits

Independent central finite difference agrees with autograd to
6.49e-11 relative. Exact-null,
weighted-projection, duplicate-affine and ensemble identities pass. Recomputed
affine predictions differ from prior sealed controls by at most
2.16e-16 before rounding.
Maximum final TRAIN weighted orthogonality residual is1.15e-18; maximum
Torch/NumPy raw-correction forward difference is8.73e-11.

The score adds1MF for affine,20MF for a single member and80MF for the ensemble
to each parent's own cold bill. These are information-screen reserves, not
qualified deployment bills. No new native FlopScope inference or submission
candidate was produced. Offline fitting arithmetic is not represented as a
metered deployment cost.

Both guarded jobs completed and retired. CPU-only Torch2.7.0/NumPy1.26.4,
one intra-op/inter-op/BLAS thread, nice10, peak observed RSS674562048 bytes,
zero additional swap. The fit took15.000s internally and
16.908s including process startup/watch. Added artifacts
were approximately2.70MiB; no full weight matrices or new datasets
were saved. Original captured inputs and every candidate/archive are preserved.

## Consequence for the active goal

Weight5 remains fully qualified locally at3.883207214854186e-9 NC on existing14,
with its exact archive approval pending and no upload receipt. The verified
official best stays#3315844.015535762022727e-9; no official4e-9 bank yet and
approximately1e-9 remains unfinished. This negative adds evidence against
further unconstrained terminal refits on this representation.

A separate source check found that exact raw-third-moment recentering and
persistent full-covariance GMM4 were already run and failed their fixed tests;
they are not newly available experiments merely because a design file exists.
See `astra_phase2_reexamination_v1/THIRD_RECENTER_ROUND_V1_CLOSED.md` and
`POSITIVE_GMM4_SPLIT_MERGE_V1_CLOSED.md`. Neither was rerun in this turn.
