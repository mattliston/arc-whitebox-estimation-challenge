# One fixed full-carrier joint prefix/final fit

September 6, 2026, 11:20 PDT. Frozen before this track opens training targets.
Phase 2 only. No existing artifact, candidate or payload is modified.

## Hypothesis and lineage

Jointly adapting the existing prefix encoder/messages/mean heads and its
actual final consumer may escape the frozen-consumer limitation of the
earlier single-gradient encoder ray and frozen-encoder readout pilots.
The complete dynamic CP/star/trace state is recomputed on every forward;
there is no teacher forcing, target-conditioned feature or frozen-K3 shadow.
Use the passing joint VJP and finite implementation certificates in
JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md. All old negative receipts stand.

Start from payload_compact_feed_fullpair_v1.npz (P) and
payload_fullpair_final_head_v1.npz (Q). Train exactly the previously certified
78,097 parameters. Centers, scales, rms and first remain frozen. No new
inference operations, larger network, moment coefficients or selector rules.

## Data and fixed split

Ranks 35 through 234 in the sealed_target_blind_feature_store_v3 observer_fit
role. Resolve weights through manifest.json and check row-manifest hashes.
Do not duplicate the 200 weight files. Targets are the documented 1e9-sample
post-ReLU means in public_g22_operator_features_v1/truthpost/truthpost_rankNNNN.npz,
key mc_post. These are public development nets, NOT pristine holdouts.

Reuse the existing final-head 160/40 split. Validation ranks:
37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,
224,227,233,234. No validation target is opened by the fit. Save final
validation predictions once; the separate NumPy judge opens those targets
only after the fixed terminal checkpoint has been written.

## Fixed optimizer, objective and endpoint

Six complete epochs, 160 training nets each, batches of four nets with
immediate per-net backward and gradient accumulation (one graph resident).
Shuffle using PCG64DXSM seed 930696; no data augmentation. Final-layer MSE
plus 0.05 times the mean MSE of layers 0..14, divided by the fixed 2e-8
normalizer. The auxiliary term limits destructive prefix drift; it is not
the competition score. Nonnegative output clipping matches deployment.

For each trainable tensor theta, parameterize theta = theta0 + s*z, with
s = max(original tensor RMS, 0.001), and z initially zero. AdamW on z,
lr 0.002, betas (0.9,0.999), eps 1e-8, weight_decay 0; cosine decay to
0.0002 over the fixed 240 updates. Clip the concatenated z gradient norm
to 5. After each update, project each z tensor to RMS <=0.10. This fixed
relative trust region is not tuned to training or validation outcomes.
Record clipping/projection and group displacement. Never choose an earlier
checkpoint by its score; epoch snapshots are audit/recovery artifacts only.

Abort on nonfinite loss/gradient/parameters/predictions or resource failure;
preserve the failed run and do not retry with another hyperparameter.
No accuracy-based early stopping. Report online training losses, but do
not change the schedule. No validation or panel fitting during the run.

## Independent judgment and release gates

Replay original P0/Q0 and terminal P1/Q1 through the actual NumPy reference
on all 40 validation nets. Compare the saved MPS terminal predictions with
NumPy (maximum all-layer difference <=3e-6 on every net). Require pooled
final raw MSE ratio <=0.90 and at least 24/40 individual raw improvements
before broadening to the existing 14-net grading panels. This is a stage
gate, NOT evidence for a particular adjusted-score milestone. If failed,
close this prescribed fit without optimizer/epoch/radius sweeps.

Any submission still needs exact instrument/candidate payload parity, both
public8 and virgin6 panels, actual flopscope.12 billing, persistent Whest.16
total/residual wall time and memory, validation, immutable package hashes,
and the user's requested milestone bank threshold. No forum posts.

## Resource and implementation checks

One nice10 MPS worker, at most three CPU threads; never import threadpoolctl
in the AP Torch interpreter; MPS fallback disabled. Load only one net's
weights at a time and release each graph. Before opening targets, certify
the new manifest-backed card constructor on ranks35/100 against the exact
old weight bytes, fixed features, predictions and selection memberships.
No other MPS job while the fit is alive. CPU helper agents stay small.
