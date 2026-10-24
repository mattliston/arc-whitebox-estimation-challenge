# Learned response-conditioned pair memory v1 — completed, gates failed

Phase 2 only. Written September 6, 2026, 17:04 PDT, using the UTC clock
and result-file mtime rather than inherited handoff dates. The one paired
fit finished at 2026-09-06T23:59:24.819136Z (16:59:24 PDT). Both arms
completed all 100 epochs. The process returned exit 1 because the fixed
information gates failed, not because training was interrupted or timed out.

## Exact experiment and scope

The frozen protocol is `LEARNED_RESPONSE_PAIR_MEMORY_PROTOCOL_DRAFT_V1.md`,
SHA256 `1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c`.
Root acceptance is `LEARNED_RESPONSE_PAIR_MEMORY_ROOT_ACCEPTANCE_V1.md`,
SHA256 `45c590e373fb35c6af0c9a248addf8b98d5e6830c8b6b66861fd5ced3b537e2a`.

This was one 10-parameter response-unit-correct recurrent pair field,
compared with an independently trained no-history RESET control. TRAIN
was ranks 35..58 excluding development VALID ranks 37, 39, 42 and 48:
20 TRAIN and four held-out development networks. They are not a new,
pristine 14-net panel. Both arms used seed 930074, zero initialization,
100 epochs, Adam lr 0.001, ridge 0.001 and gradient-norm cap 5, without
checkpoint selection or a hyperparameter sweep.

The lower-order trajectory and past post K21 planes were teacher-forced.
Only the latent H history was model-generated. The query was post12 to
pre13, supervised against the existing bank's missing directed pair
action. The consumers were its covariance response and a leading
Gaussian next-layer variance-to-mean response. These losses are NOT true
raw final-layer MSE, an autonomous rollout or an adjusted score.

## Fixed held-out verdict

All values below pool equal-sized coordinates and the same four VALID
networks, using the frozen final model of each arm.

| Required gate | Measured | Threshold | Verdict |
| --- | ---: | ---: | --- |
| History covariance error / zero correction | 0.9036964070131149 | <= 0.50 | FAIL |
| History next-mean proxy error / zero correction | 0.9911990674534291 | <= 0.50 | FAIL |
| History next-mean proxy error / RESET | 1.000236776947614 | <= 0.80 | FAIL |
| History beats RESET in next-mean proxy | 1 of 4 | >= 3 of 4 | FAIL |

Pooled covariance MSE: history 6.826706406415215e-8, RESET
6.826881298375888e-8, zero correction 7.554203329167537e-8.
Pooled next-mean proxy MSE: history 2.2013042803480336e-8, RESET
2.2007831856229816e-8, zero correction 2.2208498298970208e-8.
History therefore reduced this proxy error by 0.8801% versus zero, but
was 0.02368% worse than RESET. The sole history/RESET win was rank 37.
The covariance proxy's 9.6304% reduction did not translate into a useful
next-mean response gain and does not qualify the model for a rollout.

## Numerical, chronology and resource receipts

The synthetic n7 and rank35 full-width output/gradient certificates
passed before training. Rank35 and the other 23 prepared rows had
separate completed global provenance seals. The trainer reports source
and input hashes unchanged. Both final model files were sealed before
VALID target metadata was decoded; all VALID predictions were sealed
before held-out loss arrays were decoded. No validation-curve selection
was performed.

Both arms recorded nonzero gate gradients. Maximum gate-gradient norms
were 0.0190923810 for history and 0.00956499204 for RESET. History had
10 active parameters; RESET had eight active of 10 allocated, with its
two structurally inactive coefficients remaining exactly zero. This is
a completed negative result, not a claim that optimization was stuck
because gradients were detached or identically zero.

Combined fit wall time was 107.7840105 seconds against a 600-second cap;
total process wall time was 110.1884056 seconds. Reported peak process
RSS was 930,906,112 bytes against a 6 GiB cap. MPS-ready allocation was
2,708,684,800 bytes and driver memory 3,246,882,816 bytes, reported
separately from process RSS. One nice-10 MPS worker ran, with no
concurrent bank replay. Post-run system swap remained 816.75 MiB used.
These are local pilot resources, not native estimator runtime or cost.

Frozen artifact SHA256s:

- Trainer: `94dfe200f78f27f38258ffe3ad80de3b0c494fddebcfa61e7e69e7de4f20f798`.
- `learned_response_pair_memory_training_v1/result.json`:
  `99c14e4f428bfacf2dca32961e5a005ed841069afeea8c3f5ee2904cb3876687`.
- `both_models_sealed.json`:
  `b7f74d22f0c1b7bcb91531a08812fdff2dca5de50d2408cd33d0e44e052e1886`.
- `valid_predictions_sealed.json`:
  `e8d9c8f3229b3889b6328f042102448587545502d03c18502ecc2699c526d28a`.
- `history_model.npz`:
  `74005f7c672fb85abc6d7a45de8392490294334bfe56043a79e4caf9cbaa6423`.
- `reset_model.npz`:
  `ca56711642a8ba87ee4d32cb7aa3d4ba90f3feb917498d0358fbef13c3d7d0a4`.

## Binding consequence

Close this exact state law, source, readout, optimizer and 100-epoch
paired fit. No resume, larger network, altered source, normalization,
learning-rate/seed/epoch sweep or endogenous rollout follows this
failure. It does not prove all learned third-tensor compression is
impossible. The separate source-only arithmetic census is reusable
accounting, not a measured candidate or permission to deploy this law.

The prospective 290B whole-card budget and conditional approximately
2.96e-9 adjusted score at unchanged bank raw accuracy were assumptions,
NOT results. This fit supplies no evidence that those accuracy conditions
hold. No new candidate, raw-score claim, panel run or submission was
produced. Official bank #329926 remains 4.982231460183304e-9 adjusted.
The user goal strictly below 1e-9 remains active and unachieved.
