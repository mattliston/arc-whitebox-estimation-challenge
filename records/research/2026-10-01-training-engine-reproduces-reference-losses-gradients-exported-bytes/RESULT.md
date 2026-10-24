# Trainer lane — parallel long-horizon refit (O/trainer/)

Status as of 2026-10-02 12:50 UTC: S39 (candidate 3, five changes, in it) GRADED 1.8470e-9 (#333528; final-layer MSE -0.93%
against S38; the torch estimate for candidate 3 against candidate 2 on the 50 held-out ranks was -0.70%). The pool now judges
the pieces of the lead's and the physicist's CARRIER (an extra K31 plane) on the 50, on top of candidate 3 (section 9h): bank
channel -2.35% (se 0.25, 47/50), vector form K 16 -2.06%, K 8 -0.89%, sketched form -0.41%, its one-pass form +4.5%. Running,
each logging its own line: both pieces together, the structured state alone, the full-matrix ceiling. Round 2 of the dial
solve (mix4) is queued behind them; the carrier-strength solve is parked.

VERDICT: the learned parameters are NOT optimisation-limited in the sense of the hypothesis (several percent). What exists is a
small refit gain that comes from training on the 641 networks instead of the old 160: about -0.2% after 18 updates and -0.4%
after 23 on networks nobody has trained on, possibly still growing; my guess for a full epoch is 0.5-1%.
- Pilot, VAL-A (20 nets): EMA -0.37% after 18 updates, -0.27% after 23 (se 0.17, 0.23); raw -0.36%, -0.04%.
- Pilot, 60 networks nobody has trained on (newer ranks the run had not reached): EMA -0.10% after 18 updates (se 0.11),
  -0.43% after 23 (se 0.15); 23 against 18 on the same networks -0.33% (se 0.10).
- Exact least-squares refit of the final-layer correction, scored on held-out networks: newer ranks -0.32%, old TRAIN160
  ranks -0.03%, VAL-A +0.04%; richer readouts (up to 720 numbers) give the same at 195 fit networks and 0.13% more at 320. The validation ranks are a biased judge for
  refits: the current parameters are tuned to the old 160 and were selected on the validation ranks.
- The whole learned mean correction is worth 13.5% of the final-layer MSE (3.4% of it at the final layer); the applied
  final-layer correction has cosine 0.20 with the correction that was needed.
(My 21:17 report said "no trend after update 15, stop the pilot"; the fresh-network test at 21:46 showed that was premature.)

Hypothesis under test: the estimator's learned parameters (23 arrays, 47,581 numbers, plus 10 dials) are optimisation-limited, so
thousands of optimiser steps on the 641 training networks lower the final-layer MSE by several percent with no change to the
native code or its bill.

## What exists

| file | what it is |
|---|---|
| `otr_common.py` | paths, split (641 train / VAL-A 20 / VAL-B 20), parameter decoding from a candidate source, the file queue |
| `otr_engine.py` | the torch card for one network, forward (+ backward); copied from the reference trainer lines 54-168 |
| `worker.py` | persistent worker: claims one request file, runs one network, publishes gradients or the 16-layer MSE |
| `supervisor.py` | keeps workers alive; each start goes through `O/gate.sh`; never more than 3 workers on the box; drops a worker when system swap grows; requeues the requests of a worker that is gone |
| `master.py` | owns parameters, Adam, the moving average (EMA); one JSON line per update; checkpoints; exact resume |
| `evalnet.py` | forward-only evaluation of one parameter file on a list of networks |
| `export.py` | parameters -> native `candidate.py` (only value expressions replaced) |
| `native_pair.sh`, `native_compare.py` | native probe of parent and candidate on the same ranks, paired comparison (not run yet) |
| `cert_reference.py`, `certify.py` | certification against the reference trainer |
| `bench_grad.py`, `cert/bench_run1.log` | gradient time against thread count and memory relief |
| `grad_stats.py`, `drift_stats.py`, `prequential.py`, `ema_at.py` | diagnostics on the gradient files and parameter files of a run (sections 4 and 5) |
| `probe_capture.py`, `probe_fit.py` | final-layer headroom probe: capture what the message network sees, exact least-squares readouts (section 5) |
| `line_search.py` | VAL-A line search along the average training gradient; written, NOT run (the gradient turned out to be noise) |

Run directories: `run_cert/` (certification), `run_smoke/` (3-update plumbing test), `run_A/` (the pilot).
Nothing is ever deleted or overwritten: every per-network gradient (`run_A/results/g<update>_<i>_r<rank>.pt`), every parameter
set (`run_A/params/p<k>.pt` = parameters after k updates, `ema<k>.pt`) and every checkpoint stays on disk.

## 1. Certification (all passed, `CERTIFY.json`)

All with the untrained parameters of the research parent RRPI (`H/hill_climb_pair_rr_inverse_v1/candidate.py`, sha a3efb3a1...), 1 thread.

- (a) Final-layer MSE of training ranks 49, 209, 214, 117 from my worker in gradient mode against the first four rows the
  reference trainer logged (`H/hill_climb_rrpi_detached_star_v1/segment_00/steps.jsonl`): identical to all 17 printed digits
  (1.4424029867226309e-08, 1.649986769412125e-08, 1.7278360746907035e-08, 1.701267378410074e-08). The forward-only path
  (no_grad, used for validation) gives the same four numbers bit for bit.
- (b) Gradient of rank 49 (loss scaled 1e8/4): my worker against the reference trainer's own statements executed in a separate
  process (`cert_reference.py` runs the reference source up to its optimiser set-up, leaving out only the 9 ceremony statements
  that read sys.argv or create its segment directory). All 23 arrays and 10 dials are bit-identical (largest relative difference
  0.0). The four gradients added in network order give exactly the two norms the reference logged for its update 1:
  7.338744163513184 (arrays) and 4.599334701658686 (dials). The reference's sealed file pins all still hold.
- (c) Exporter: fed the untrained parameters it reproduces the RRPI source byte for byte (same sha256). Fed the reference
  trial's 8-update endpoint it produces a file whose decoded arrays and dials equal the endpoint bit for bit and whose syntax
  tree equals the reference exporter's candidate (the two files differ in one line: the reference exporter dropped a comment).
- Extra: exact resume. In `run_smoke/` the master was restarted from the checkpoint after 2 updates; it replayed update 3 from
  the gradient files on disk and reproduced the parameter file and the checkpoint (parameters, EMA, Adam state) bit for bit.

## 2. Baseline (`baseline_valid40.json`)

Untrained RRPI parameters, torch card, forward only, 1 thread, against the 1e9-sample truth (`fable_refit_v1/truth688`).
The torch card approximates the native estimator; the native probe is the judge.

| set | networks | pooled final-layer MSE | pooled all-layer MSE |
|---|---|---|---|
| all 40 validation ranks | 40 | 1.695523e-8 | 6.654e-9 |
| VAL-A (even positions; checkpoint selection) | 20 | 1.738865e-8 | |
| VAL-B (odd positions; report only, never used for choices) | 20 | 1.652182e-8 | |

Spread across the 40 networks: sd 2.28e-9, min 1.289e-8, max 2.190e-8. The pilot's own VAL-A evaluation at 0 updates reproduces
the 20 VAL-A numbers of this file bit for bit.

Does a torch-card change predict the native change? Checked on the one trained endpoint that has native numbers, the earlier
team's 8-update refit (`H/hill_climb_rrpi_detached_star_v1`, measured natively on all 40 validation networks in
`H/hill_climb_rrpi_detached_star8_component40_v1/closeout.json`): native pooled raw change +0.053% (their interval -0.15..+0.26),
my torch card on the same 40 networks +0.011% (`cert/torch_valid40_reference_endpoint8.json`); per-network changes correlate 0.80,
root-mean-square difference 0.44% per network. So a pooled torch change on 20 networks is trustworthy to about 0.1%.
Their conclusion at the time: 8 updates gave nothing on 40 networks, 16 updates gave -0.31% on 8 networks (interval crossing
zero), branch closed.

## 3. Throughput (the 26 s figure does not reproduce today)

Measured 2026-10-01 on ranks 49 and 209, one process at a time, nice 10 (`cert/bench_run1.log`):

| threads | wall per network gradient | user + system CPU | peak resident memory | equal to the 1-thread reference |
|---|---|---|---|---|
| 1 | 33-37 s | 22-24 s + 11-13 s | 6.5 GB | bit-identical |
| 1, memory relief off | 34-36 s | 23 s + 12 s | 8.8 GB | bit-identical |
| 4 | 16-19 s | 27-30 s + 14-18 s | 6.6 GB | MSE differs in the 5th digit; gradient within 3.4e-4 of each array's largest entry |
| 8 | 18-22 s | 37-42 s + 27-32 s | 6-8.8 GB | as 4 threads |

In the pilot (278 gradients, 1 thread each): median 40.2 s per gradient, 10th-90th percentile 34.6-56.1 s, peak resident memory
6.74 GB; forward-only evaluation median 10.5 s, peak 3.5 GB.
Workers: the machine gate admitted 1, then 2, then 3 workers. With 3 workers system swap jumped twice (19:30 UTC +135 MiB,
19:54 UTC +261 MiB); the supervisor dropped a worker each time and from 19:58 UTC the trainer is capped at 2 workers. A third
jump (20:55 UTC, +229 MiB, when another lane started an MPS job and a second native job) dropped one of the two for 10 minutes.
Rate: 270 network gradients per hour with 3 workers (updates 1-5, 160 s each), 178 with 2 (243 s per update), about 90 with 1.
Overall the first 276 gradients took 1.99 h = 139 per hour, including 300 extra forward evaluations I inserted, the three
drops and one 9-minute stall (a dropped worker's network was not handed over; fixed 19:39).

## 4. Pilot run A (stopped after 28 updates on the lead's decision, 22:26 UTC)

Launched 2026-10-01 19:05 UTC. Batch 12 networks per update, 2 epochs of the 641 training ranks = 106 updates (the trailing 10
networks are not run), Adam with full 1e-4 / trunk 5e-5 / dials as DIAL_LR, linear warm-up over 10 updates then constant, the
batch-summed gradient (loss scaled 1e8/12) clipped to norm 1 for arrays and for dials separately, EMA decay
min(0.95, (1+k)/(10+k)), VAL-A (20 nets, forward only, raw and EMA parameters) every 18 updates, checkpoint every 25.
1 thread per worker.

- watch: `tail -F O/trainer/run_A/progress.txt` (one line per update / evaluation); machine-readable: `run_A/log.jsonl`
- workers: `tail -F O/trainer/run_A/workers/supervisor.log O/trainer/run_A/workers/w*.log`
- stop: `touch O/trainer/run_A/STOP` (workers and master leave after the current network)
- resume: move the STOP file aside, then from O/trainer/
  `(nohup /Users/matt/anaconda3/bin/python -B supervisor.py run_A 2 1 >> run_A/workers/supervisor.log 2>&1 < /dev/null &)` and
  `(nohup /Users/matt/anaconda3/bin/python -B master.py run_A --epochs 2 --batch 12 --eval-every 18 --ckpt-every 25 --ema 0.95 --warm 10 >> run_A/master.out 2>&1 < /dev/null &)`

VAL-A = 20 selection networks, torch card forward only, against the untrained RRPI parameters on the same networks; "se" =
standard error of the mean per-network change. TRAIN-A = 20 fixed training networks (`cert/train_a_ranks.txt`), same measure.

| updates | network gradients | VAL-A raw | VAL-A EMA | TRAIN-A raw | TRAIN-A EMA |
|---|---|---|---|---|---|
| 0 | 0 | 1.738865e-8 | (same) | 1.727601e-8 | (same) |
| 18 | 216 | -0.36% (se 0.19, better on 14/20) | -0.37% (se 0.17, 15/20) | -0.28% (se 0.22, 11/20) | -0.28% (se 0.19, 11/20) |
| 23 | 276 | -0.04% (se 0.28, 10/20) | -0.27% (se 0.23, 12/20) | not run | not run |

The row at 23 updates is my own evaluation of `params/p00023.pt` and of the EMA rebuilt by `ema_at.py` (bit-identical to the
master's file where both exist), made because the master's next scheduled point (36 updates) was more than an hour away.
All-layer MSE on VAL-A: after 18 updates raw +0.13%, EMA +0.07%; after 23 updates raw -0.22%, EMA -0.15% (the loss only
contains the final layer).

Fresh-network curve (`prequential.py`): in the first epoch each network's loss is computed before the run has trained on it;
compared with the untrained loss of the same network (from the probe captures, 420 of the 641 training networks):

| updates | parameters after | networks | change vs untrained (se) | newer ranks only | old TRAIN160 ranks only |
|---|---|---|---|---|---|
| 1-6 | 0-5 updates | 50 | -0.05% (0.02) | -0.05% (0.03) | -0.06% (0.04) |
| 7-12 | 6-11 | 48 | -0.06% (0.16) | -0.19% (0.19) | +0.27% (0.28) |
| 13-18 | 12-17 | 44 | -0.21% (0.13) | -0.20% (0.16) | -0.28% (0.26) |
| 19-24 | 18-23 | 45 | -0.16% (0.18) | -0.27% (0.18) | +0.21% (0.45) |
| 25-27 | 24-26 | 27 | -0.76% (0.32) | -0.76% (0.33) | (1 network) |

(table as of 22:24 UTC with 420 of the 641 training networks captured; these are the RAW parameters in use at each update,
which wander more than the EMA.)

Direct fresh-network test of two parameter sets (`cert/fresh60_ranks.txt`: 60 training networks of the newer ranks, not in
the old TRAIN160, never used for selection, not reached by the pilot before update 31; untrained loss from the captures;
`cert/fresh60_ema00018.json`, `cert/fresh60_ema00023.json`):

| EMA parameters after | change vs untrained on the 60 | better on |
|---|---|---|
| 18 updates | -0.10% (se 0.11) | 33/60 |
| 23 updates | -0.43% (se 0.15) | 40/60 |
| 23 against 18, same networks | -0.33% (se 0.10) | |

Armed for the 36-update point: `cert/fresh_at36.sh` scores the EMA after 36 updates on the 36 of these networks the run will
not have reached by then (`cert/fresh_at36_ranks.txt`), result in `cert/fresh36_ema00036.json`.

Rerun `python -B prequential.py run_A probe/untrained 6` as the pilot advances; `python -B fresh_curve.py` prints the EMA
points on the fresh networks. The last 221 training networks are not captured: I cancelled that queued process at 21:17 UTC to
keep the lane's memory use down (`bash cert/probe_rest.sh` would do it, one 3.5 GB process, about 45 min on a quiet box).
A native candidate exported from the EMA parameters after 18 updates as a dry run of the exporter (reload bit-exact, 11 value
statements changed): `candidates/pilotA_ema00018/candidate.py` (sha 8f3603d8...). Not run natively.

## 5. Diagnostics: is there anything left to optimise?

1. Gradient consistency (`grad_stats.py`, per-network gradient files of the pilot). On the first 144 training networks the
   per-network gradients are noise around zero in every array: first-order descent rate of all arrays together -0.001% of the
   training loss per update (standard error 0.026%). On the first 72 networks one array stood out, `p.enc4_w` (last layer of
   the message network): mean gradient three standard errors from zero and rank one. Tracked update by update, its projection
   fell from +0.90 (update 1) through zero (update 6) to -0.58 (update 11): a real but steep direction, used up within six
   updates of the warm-up. The rate estimator was checked on synthetic gradients with a known mean (unbiased; standard error
   right to 17%).
2. Final-layer headroom probe (`probe_capture.py`, `probe_fit.py`, captures in `probe/untrained/`). Untrained parameters. The
   final-layer correction is linear in 65 numbers (head[15], head_b[15]); solved exactly by least squares on 80 training
   networks and scored on networks not used in the fit (change of pooled final-layer MSE against the card as it is):

   | readout fitted by least squares | fit networks (80) | held-out training (40) | VAL-A (20) |
   |---|---|---|---|
   | A: the card's own 65 numbers (64 state outputs) | -0.44% | -0.29% (se 0.09) | +0.07% (se 0.14) |
   | B: 128 hidden units of the message network (129 numbers) | -0.64% | -0.31% (se 0.11) | +0.11% (se 0.20) |
   | C: 128 hidden + 126 raw inputs (255 numbers) | -0.87% | -0.14% (se 0.15) | +0.29% (se 0.23) |
   | D: C + 465 pairwise products of the 30 unit features (720 numbers) | -1.28% | +0.12% (se 0.21) | +0.48% (se 0.27) |

   With more captures and a random split (244 training networks captured; 100 held out at random; `probe_fit.py ... 40,80,140`),
   fitted on 140 training networks:

   | readout | fit networks (140) | held-out training (100) | VAL-A (20) |
   |---|---|---|---|
   | A (65 numbers) | -0.36% | -0.24% (se 0.06) | +0.02% (se 0.11) |
   | B (129) | -0.52% | -0.26% (se 0.07) | +0.03% (se 0.15) |
   | C (255) | -0.65% | -0.23% (se 0.08) | +0.03% (se 0.16) |
   | D (720) | -0.91% | -0.26% (se 0.09) | +0.00% (se 0.17) |

   So an exact refit of the final-layer readout is worth about a quarter of a percent on unseen training networks and nothing
   on VAL-A, and richer functions of the same inputs add nothing beyond it at 140 networks (at 40 networks they overfit:
   C +0.36%, D +0.55% on held-out).
   With all 420 captured training networks (`cert/probe_fit_420.txt`; 100 held out at random), change on the 100 held-out
   training networks as the number of fit networks grows (standard errors 0.07-0.18), and VAL-A at 320:

   | readout | fit on 40 | 80 | 160 | 320 | VAL-A at 320 |
   |---|---|---|---|---|---|
   | A (65 numbers) | -0.07% | -0.16% | -0.21% | -0.26% | -0.09% (se 0.12) |
   | B (129) | +0.10% | -0.10% | -0.20% | -0.31% | -0.10% (se 0.16) |
   | C (255) | +0.61% | +0.03% | -0.14% | -0.29% | -0.10% (se 0.15) |
   | D (720) | +0.86% | +0.07% | -0.20% | -0.39% (se 0.08) | -0.19% (se 0.16) |

   The card's own 65 numbers level off near -0.3%; the 720-number readout is still improving with data and at 320 networks is
   0.13% ahead of them. So richer functions of the same inputs do begin to pay with more training networks, slowly: a tenth of
   a percent per doubling of the data at the final layer.
   Why VAL-A disagrees: split the held-out networks by history (295 captures, fitted on 195, 100 held out, 10 random splits;
   mean change, spread between splits in brackets):

   | readout | held-out NEWER ranks (nobody trained or selected on them) | held-out old TRAIN160 ranks | VAL-A |
   |---|---|---|---|
   | A (65 numbers) | -0.32% (0.08) | -0.03% (0.07) | +0.04% (0.06) |
   | B (129) | -0.36% (0.08) | -0.08% (0.09) | +0.05% (0.06) |
   | C (255) | -0.30% (0.07) | -0.02% (0.13) | +0.13% (0.08) |
   | D (720) | -0.34% (0.07) | -0.11% (0.14) | +0.07% (0.08) |

   The current parameters are tuned to the 160 old training ranks and were selected on the validation ranks: on those a refit
   gains nothing, on fresh networks it gains a third of a percent at the final-layer readout alone. The card's own error does
   not differ between old and newer ranks (1.681e-8 vs 1.717e-8, se 0.04e-8 and 0.02e-8), so this is tuning, not a different
   kind of network. Consequence: valid40 understates what a refit is worth on the board's fresh networks.
   A fitted on each VAL-A network by itself gives -8.7% in-sample where pure noise alone would give -6.3%: the residual is
   almost unpredictable from what the message network sees. Per network the applied final-layer correction has cosine 0.20
   (sd 0.05) with the needed one and carries 3.7% of its power; a per-network best multiplier would add 0.23% (in-sample).
3. What the learned mean correction is worth (forward only, VAL-A, untrained parameters otherwise):
   switched off at the final layer only: +3.4%; switched off at every layer (head, head_b = 0): +13.5% (worse on 20/20;
   `cert/vala_headscale_0.00.json`); scaled by 0.9: -0.18% (se 0.13%); by 0.8: -0.02% (se 0.25%).
4. Does the torch card predict native changes? See section 2: yes, to about 0.1% pooled over 20-40 networks.

Reading: the correction already sits at the least-squares optimum of what its inputs can predict. What is left from re-fitting
to the current card and the 641 networks is of order 0.2-0.4%, which is what the pilot's first point shows. Several percent
would need new information in the inputs of the learned parts, not more steps.

## 6. Run L (the long run), launched 2026-10-01 22:28:50 UTC on the lead's GO

Purpose: answer "does continued training keep paying on fresh networks?" with a validation set that stays fresh.
Run L continues pilot A's checkpoint after 25 updates (`run_A/ckpt/after_00025.pt`: parameters, moving average, Adam state),
trains on 541 networks (`cert/runB_train.txt`) and keeps 100 newer-rank networks out of training for good
(`cert/runB_val.txt`: not in the old TRAIN160, not in valid40, not a panel network, never reached by pilot A; the same list
is the lead's set for native judging of learned-parameter candidates). Batch 12, same learning rates, no warm-up, 4 epochs =
180 updates, checkpoint every 25. The moving average (decay min(0.98, (1+k)/(10+k)), k = updates since the parent) is scored on
the 100 every 18 updates; the untrained parameters were scored once as the reference. 2 workers x 4 threads; evaluations run at
4 threads too (reference and candidates in the same frame). Update counts in the progress lines include pilot A's 25.

Launched with (from O/trainer/; the same two commands resume it after a stop):

    (TRAINER_PL=1 nohup /Users/matt/anaconda3/bin/python -B supervisor.py run_L 2 4 >> run_L/workers/supervisor.log 2>&1 < /dev/null &)
    (nohup /Users/matt/anaconda3/bin/python -B master.py run_L --init run_A/ckpt/after_00025.pt --train-file cert/runB_train.txt \
        --val-file cert/runB_val.txt --eval-what ema --epochs 4 --batch 12 --warm 1 --ema 0.98 --eval-every 18 --ckpt-every 25 \
        >> run_L/master.out 2>&1 < /dev/null &)
    (nohup /Users/matt/anaconda3/bin/python -B announce.py run_L 25 >> run_L/announce.out 2>&1 < /dev/null &)

- watch: `tail -F O/trainer/run_L/progress.txt`; one line per held-out evaluation also goes to `O/TEAM_LOG.md` (`announce.py`:
  change against untrained, standard error, networks better).
- stop: `touch O/trainer/run_L/STOP`.
- bank candidates (lead's rule): the moving average at the first evaluation with a held-out change of -0.6% or better and a
  standard error of 0.1% or less, then one more per further -0.5%; exported by `announce.py` with the certified exporter to
  `candidates/run_L_ema_u<updates>/candidate.py` and announced in the team log.

| updates since the parent | network gradients | moving average on the 100 held-out networks vs untrained | better on |
|---|---|---|---|
| 25 (run L's parent = pilot A's checkpoint) | 300 | -0.237% (se 0.109) | 55/100 |

Speed (22:58-23:20 UTC, box saturated): 2 workers x 4 threads 103-163 s per network gradient = 55 per hour (13 minutes per
update); 1 worker 85-89 s = 41 per hour with a tenth of the kernel's compress/decompress traffic; memory relief off no change.
Quiet-box figure: 16-19 s per gradient at 4 threads. Also tried and dropped: workers at nice 10 instead of 15 (no difference: 112-147 s
against 121-144 s side by side) and the TBB caching allocator preloaded (`cert/bench_tbb_run1.log`: 198 s wall, four times the
page faults, and results no longer equal to the reference in the 5th digit). The limit is the box's memory: the kernel
compresses and decompresses 0.7-1.2 GB per second while two workers, two native jobs, a physicist screen and Firefox run. `echo 1 > run_L/WORKER_COUNT` (or 2) changes the worker count without a
restart. The lead's bar "standard error <= 0.1%" cannot be met with 100 networks (0.109% at the start); `announce.py` logs a
pass of -0.6% without exporting until the lead restates the rule.

`cert/master_as_launched_for_run_A.py` is the master as it was when pilot A started; the current `master.py` builds the same
configuration for pilot A's command line.

## 7. Per-layer physics dials (lead's task of 20:45 / 21:20 / 22:05 UTC): certified, gradients collected, nothing trained

`otr_pl.py`: each family is a vector of 16 zeros ADDED to today's value where the card reads it (effective value at layer l =
today's value + EPS[family][l]). With the zeros in place the card is unchanged; the gradient with respect to each zero is the
gradient of a per-layer dial started at today's global value. 217 active scalars in 16 families:
k4 (layers 1-14, today the three bands; layer 15, today 1), bscale (0-14), s_carry (1-14), s_new (0-14), c0 and c1 (1-14, shared
by the pair map and the covariance-output remainder), iso / k21n / k31 (2-14, today -0.094 / -0.028 / -0.212), edge (8-14,
0.164), rc (1-14, -0.08), X0 / X1 / X2 (birth legs; X3 has no effect, the j3 leg is cut), kappa (birth weight
d (1 + kappa radial/3), today 1), bsrc (source term of the b carrier, today 1).

Certificate (`pl_certify.py`, `cert/pl_certify_run1/pl_certify.json`, `cert/pl_certify_run1.log`; ranks 49, 209, 214, 117,
untrained RRPI parameters, 1 thread, against the certified worker results of `run_cert/`):
- final-layer MSE bit-identical on all four networks;
- gradients of the 23 arrays and 10 global dials bit-identical (largest relative difference 0.0);
- every per-layer gradient finite;
- per-layer gradients summed over the layers a global dial covers equal that dial's gradient to 2e-15 relative or better.

On those four networks 20 of the 217 scalars have the same sign on all four; chance alone gives 27. Nothing beyond noise.
Because the leaves change no bit of the losses or of the standard gradients, run L's workers run with them switched on
(`TRAINER_PL=1`, since 22:48 UTC): every gradient file `run_L/results/g*.pt` also holds `pl_grads` (per family, 16 values).
`python -B pl_stats.py run_L <first> <stop>` gives mean, standard error and sign consistency per scalar over the networks.

### 7a. Gradients over 72 networks and forward scans (2026-10-02 00:50-02:15 UTC)

`pl_stats.py run_L 0 6`: over the first 72 training networks of run L, 39 of the 217 scalars have a mean gradient more than
2 standard errors from zero (noise alone: 10) and 11 more than 3 (noise alone: 0.6). t = mean / standard error:

| family | what stands out | first-order worth |
|---|---|---|
| bsrc (b-carrier source scale) | sum over layers t +4.3, no layer structure beyond the common shift | -1.0% of the loss per -0.1 |
| rc (recentring strength) | sum t +4.9, every layer positive | -0.18% per -0.1 |
| c0, c1, k31 (K31 row factor / remainder) | sign change with depth: layers 1-2 t -3.3..-3.7, layers 7-12 t +2.5..+3.2 | 0.2-0.3% for 0.1 per layer |
| bscale | layers 6, 7, 14 t +2.7..+2.9, sum t +2.4 | -0.44% per -0.01 |
| X1 | sum t -2.9 | -0.52% per +0.1 |
| k4 (15 layers), iso, k21n, edge, kappa, X0, X2, s_carry | nothing beyond noise | |

Forward scans (`pl_scan.py`, `pl_scan2.py`; 16 newer-rank networks no run has trained on, `cert/scan16_ranks.txt`; run L's raw
parameters after 6 updates; 4 threads; change of pooled final-layer MSE against no shift on the same networks):

| setting (all layers) | change | standard error | better on |
|---|---|---|---|
| b source x 0.9 | -0.60% | 0.29 | 10/16 |
| b source x 0.8 | -0.59% | 0.39 | 10/16 |
| b source x 0.7 | -0.40% | 0.59 | 7/16 |
| bscale -0.01 | +0.26% | 0.21 | 7/16 |
| bscale -0.02 | +0.98% | 0.29 | 3/16 |
| bscale -0.04 | +2.09% | 0.40 | 1/16 |
| recentring rc -0.1 (0.92 -> 0.82) | -0.01% | 0.07 | 10/16 |

Reading: the b-carrier SOURCE (radial d v / 6) wants a scale of about 0.85 (parabola through the three points: minimum at
0.85, -0.67%); the carried-b scale is at a sharp optimum (lowering it hurts although its gradient suggested otherwise: the
curvature is about 2300% per unit squared); recentring is flat. The gradients overstate what a finite step delivers.
The b-source scale is a new dial: natively it is the constant 6 on line 3397 of the RRPI source (`export.py` writes it as
`_S19_BSRC_DIV = 6 / scale`, two substitutions on that line and one new literal; same operations).
Machinery ready, not used for training yet: `master.py --extra bsrc:0.004,rc:0.002` makes families GLOBAL trainable dials
(shifts travel in the parameter files as `pl`; workers apply them; dry-tested at start only); `--init-updates` sets the update
count behind an `--init` checkpoint of a run that itself continued another.

Native lines that read each dial (RRPI source, effective definitions): k4 band 3335-3336; bscale 3400-3401; s_carry / s_new
3463-3467; K31 row factor `_K31R_TABLE` 6722, read in `_k31_row_scale` 7725-7730 (already per layer; `_gn_dcov` 7607 uses the
same function); iso / k21n / k31 `_GN_KNOBS` 3919, read in `_gn_dcov` 7595 / 7604 / 7609; edge `_GN_EDGE` 3921 (read 3951-3954)
and rc `_GN_RC` 3922 (read 3380-3382), both already per layer; X1 `_BD_J1_PHYS` 7663 / 7673; X0, X2 `_S19_BD_ADD[:3]`
7650 / 7678; birth weight 3396; b source 3397. Line 7807 pre-binds the scalar constants iso, k31 and the c1 values at setup:
per-layer values multiply the number of distinct constants.

## 8. Run L's end, and the b-source scale on the held-out networks (2026-10-02 02:15-03:10 UTC)

Run L was stopped for good at 03:09 UTC after 6 updates (72 network gradients) on the lead's decision: the card it trains is
replaced by the card with F4 + K4L (section 9). It ran at 12-20 minutes per update because the box was saturated. It wrote no
checkpoint of its own; its parameters after k updates are `run_L/params/p0000k.pt`, its start is `run_A/ckpt/after_00025.pt`.
What it leaves: the per-layer gradient table of section 7a (72 networks), the held-out reference numbers below, and the
evaluation list `cert/runB_val.txt` (100 newer ranks no run trained on), which is now the team's judge for learned parameters.

Held-out evaluation of the b-source scale (torch card WITHOUT F4 / K4L, forward only, 4 threads; 97 of the 100 held-out
networks were done when the workers were paused for the lead's quiet window; the same 97 networks in every row;
`cert/heldout_cand.py cand2_ema25_bsrc085`):

| parameters | change of pooled final-layer MSE against untrained | standard error | better on |
|---|---|---|---|
| moving average after 25 updates (run L's start) | -0.234% | 0.112 | 53/97 |
| the same + b-carrier source x 0.85 | -0.808% | 0.210 | 68/97 |
| the scale alone (second row against the first) | -0.576% | 0.155 | 63/97 |

Candidate file with the one changed native line: `candidates/ema25_bsrc085/candidate.py` (sha256 d3f87dd1ed9e...; not run
natively). It was measured on the old card: F4 and K4L act on the same fourth-order feed, so the scale has to be re-measured on
the new card; in run M it is a trainable dial.

VOID EPISODE (kept on disk, `run_L/void_20261002_0230/README`): the first held-out evaluation of that candidate ran on a worker
that had been started before the worker code learned to read shifts from a parameter file, so it returned the plain moving
average (the give-away: a standard error of exactly 0 against it after 30 networks). The files were moved aside, the worker
restarted, the shift path proven on network 629 (shift from the parameter file = shift from the request, bit for bit), and the
evaluation repeated under a new label. Since then every result carries the card it was computed on and the master refuses a
result from another card (section 9).

QUIET WINDOW MISSED: the lead announced a quiet window at 02:27 UTC in TEAM_LOG and by message; messages reach this session
only between turns and the log was not being re-read, so run L kept two workers running until the lead paused them at 02:55.
Habit since: read the `[lead]` lines of TEAM_LOG before every launch.

## 9. Run M: the card with F4 + K4L, strengths trainable (written and certified 2026-10-02 03:10-03:40 UTC; NOT RUNNING)

Lead's instructions (01:00, 01:38, 03:17 UTC): train on the card of the S37 file, i.e. today's card + the physicist's F4
(rule A, cheap anchor) + the lead's K4L, with the F4 strength theta per column age (1 and 2 layers, start 0.6), the K4L
strength c per pair age (1 and 2, start 1), the b-source scale and the recentring strength as trainable dials, and the other
consumers of the off-diagonal fourth cumulant free to move with them (k31 knob, a k22 scale).

| file | what it is |
|---|---|
| `otr_m.py` | F4 + K4L on the card, each strength = base + a per-layer leaf (families `f4a1`, `f4a2`, `k4la1`, `k4la2`); `k22` = scale of the off-diagonal part of the (2,2) plane given to the pair map. Installed after `otr_pl.py`'s hooks, so K4L acts after the k4 band dial and F4 after the column-weight dials, as in the two reference modules and in the native file. |
| `otr_engine.py` | `TRAINER_CARD=M` switches the card on (needs `TRAINER_PL=1`); every result carries `card` and `star_attach`. `STAR_ATTACH`: the rank-32 star basis differentiable in the backward pass (same forward values). |
| `master.py` | `--card M` (a result from another card stops the run); `--extra` families as global trainable dials; the file `<run>/STAR_ATTACH` (1) makes the gradient requests of the following updates use the attached basis. |
| `m_certify.py` | forward certificate, one network per process (`cert/m_certify/r<rank>.json`) |
| `m_gradcheck_inproc.py`, `m_gradcheck_table.py` | gradients against central finite differences in one process (`cert/m_gradcheck.jsonl`); `m_gradcheck.py` is the same through a worker pool |
| `announce.py` | now names the run it reports; an exported candidate gets `physics_dials.json` (every extra dial: shift, base, value) next to it for the engineer's build |
| `cert/*_as_run_L.py` | engine, worker, master, supervisor, per-layer module, exporter and announcer as run L used them |

Forward certificate (networks 35 and 100, 4 threads, per-layer MSE of all 16 layers compared to the last bit):

| check | 35 | 100 |
|---|---|---|
| run M card with theta = 0, c = 0 against today's card | equal | equal |
| run M card with theta 0.4 (both ages), c 1 against the physicist's `f4.install` + the lead's K4L hooks lifted from `e16b_k4_closed_loop.py` | equal | equal |
| run M card with theta 0, c 1 against the lead's K4L hooks alone | equal | equal |
| final-layer MSE, theta 0.4 + K4L against today's card (the lead's published -7.43% / -3.16%) | -7.432% | -3.155% |
| final-layer MSE, K4L alone (the lead's -2.69% / -3.65%) | -2.700% | -3.652% |
| final-layer MSE, the run M card at its base (theta 0.6, c 1) | -7.917% | -1.988% |

A forward run takes 26-34 s at 4 threads with two such processes on a busy box.

Planned run (one command each for the pool and the master, once the launch is allowed):
`TRAINER_PL=1 TRAINER_CARD=M supervisor.py run_M 3 4` and
`master.py run_M --epochs 4 --batch 12 --seed 0 --warm 1 --ema 0.98 --eval-every 18 --ckpt-every 6 --init run_A/ckpt/after_00025.pt
--train-file cert/runB_train.txt --val-file cert/runB_val.txt --eval-what ema --card M
--extra f4a1:0.004,f4a2:0.004,k4la1:0.008,k4la2:0.008,bsrc:0.004,rc:0.002,k31:0.002,k22:0.004`.
The first thing the master does is score the untrained parameters and the starting moving average on the 100 held-out
networks ON THE NEW CARD: that is the reference every later number is compared with, and against run L's stored numbers of
the same networks on the old card it is also the torch measurement of F4 0.6 + K4L on 100 networks.

NOT RUNNING: the start of the worker pool was refused by the permission system of the trainer's session at 03:27 UTC
("interference with other workloads"). The lead's instruction is not the user's consent; the launch waits for Matt.

### 9a. The gradient convention of every refit so far is wrong for the dials that act through the carried b (03:30-04:10 UTC)

Every refit since the star-Nystrom basis (the reference trainer, pilot A, run L) backpropagates with the rank-32 star basis
held constant ("detached star"). The lead asked for the F4 strength's gradient to be checked against finite differences because
the physicist had seen autograd at 0.58 of the difference. What the check found is larger than that.

Star alone (`star_derivative_probe.py`, `cert/star_derivative_probe_r0035.jsonl`; network 35, the stars carried out of layers
4 / 8 / 12 on the run M card, a fixed random linear read-out of the pushed star; change = unit direction x the star's norm):

| direction of the change | basis held constant | basis differentiated | central differences, relative steps 1e-4 .. 3e-2 |
|---|---|---|---|
| b x v^T (what the carried b or F4's anchor does), layers 4 / 8 / 12 | -0.172 / +0.075 / -0.081 | -0.445 / +0.464 / -0.518 | -0.445 / +0.464 / -0.518 |
| random rank one, layer 4 | -0.0003 | +0.0443 | +0.0442 .. +0.0446 |
| random Gaussian, layers 4 / 8 / 12 | +0.0042 / +0.0016 / -0.0010 | +0.0044 / +0.0049 / +0.0018 | +0.0044 / +0.0049 / +0.0018 |
| the star itself | equal | equal | equal |

The differentiated basis (plain autograd through the range finder's two QR steps, float32) is the exact derivative, valid for
steps up to several percent of the star's norm. The constant basis misses 60-85% of it in low-rank directions.
The star keeps 81-91% of its energy in 32 directions; its 32nd and 33rd singular values are within 1-2% of each other.

Whole card, network 35, derivative of 1e8 x final-layer MSE per unit of the dial (`m_gradcheck_cardL.py`,
`m_gradcheck_inproc.py`, `cert/m_gradcheck.jsonl`; parameters = pilot A after 25 updates):

| card | dial | central difference (step) | basis held constant | basis differentiated |
|---|---|---|---|---|
| today's (no F4, no K4L) | bscale | -0.48 (0.01) | +2.11 | -0.45 |
| today's | b source | +0.079 (0.1) | -0.102 | +0.040 |
| today's | recentring | +0.076 (0.1) | +0.079 | +0.080 |
| run M | bscale | -1.06, se 0.32 (line through 11 runs within +-0.004); -1.65 (0.01) | -0.54 | -0.83; with birth data in the graph -0.98 |
| run M | b source | -0.17 (0.1) | -0.26 | -0.07; -0.10 |
| run M | F4 strength, both ages | +0.040 (0.1) | +0.038 | +0.019; +0.021 |
| run M | recentring | +0.052 (0.1) | +0.051 | +0.052 |
| run M | K4L strength, both ages | -0.021 (0.05) | -0.019 | -0.019 |

Reading. On today's card the constant basis gives the WRONG SIGN for bscale and for the b source; the differentiated basis
agrees with the differences. On the run M card the differences are too noisy to rank the rules for the F4 strength: one
network's forward run jumps by about +-0.25% for parameter changes of 0.001 (`cert/finegrid_bscale_r0035.jsonl`: 11 values of
bscale between -0.004 and +0.004; neighbouring values differ by up to 0.5% in either direction), which is the known
per-network precision noise, so a central difference with step 0.1 carries about +-0.013 for the F4 strength and +-0.2 for
bscale, and small steps are useless (the first version of the check used steps of 0.002-0.05 and measured noise).

Learned arrays: the two rules give the same gradient (cosine 0.9999 on network 35; trunk and head arrays above 0.999), so
sections 4-6 stand for them. They differ for the birth-coefficient arrays `a` (cosine 0.91) and `c` (0.53) and for the dials
bscale, X0, X1, s_carry, the b source, F4 age 1 and k22. So: the dials were steered by a wrong gradient in every refit;
the per-layer gradient table of section 7a is void for bscale, the b source, X1 and s_carry (its forward scans stand, and
they had contradicted the table for bscale and the recentring).

Birth data. The two reference modules keep F4's birth scalar and K4L's hub entries and coefficients as constants. The forward
value of K4L's two legs does not depend on the hub entries, but with the entries constant the backward pass sees the pushed
columns move with them: a spurious term. `otr_m.py` with `TRAINER_BIRTH_ATTACH=1` carries the birth data in the state tuple
(a 15th element taken off again by a `state_in` hook), so they stay in the graph across the per-layer checkpoints; forward
values are unchanged (same MSE to the last bit in all four gradient runs of network 35).

k22: +-10% of the off-diagonal (2,2) plane multiplies network 35's error by 6.6 and 3.1 (it drives the recursion of the
isotropic level); the gradient puts its optimum 0.0002 from 1. Not a usable dial; left out of run M.

K4L's strength on network 35: gradient -0.019 per unit of c (the error falls 1.4% per unit), central difference -0.021.

Run M will use the differentiated basis and the birth data in the graph (`<run>/STAR_ATTACH` = 1, `TRAINER_BIRTH_ATTACH=1`).
Running since 04:06 UTC in one process: the exact gradient of every dial on 16 newer training networks at the S37 parameters
(`m_gradtable_inproc.py`, `cert/m_gradtable.jsonl`), to be followed by forward scans on the held-out ranks (`m_scan_inproc.py`).

### 9b. The exact gradient of every dial on 16 newer training networks (04:06-04:47 UTC)

`m_gradtable_inproc.py`, `cert/m_gradtable.jsonl` (table: `python -B m_gradtable_inproc.py cert/m_gradtable.jsonl --table`).
Untrained parameters (what the S37 file carries) on the S37 physics (F4 0.6, K4L 1), star basis and birth data in the graph;
networks 248 ... 603 (`cert/gradtable16_ranks.txt`: training ranks above 230, none in the held-out list); gradient of
1e8 x final-layer MSE, mean final-layer MSE 1.697e-8; 77-186 s per gradient in one process of 3 threads on a loaded box.

| dial | mean gradient | standard error | t | reading |
|---|---|---|---|---|
| k21n (covariance-output knob, rows outside the kink; -0.028) | +0.482 | 0.095 | +5.1 | more negative |
| edge (message edge strength; 0.164) | +0.016 | 0.005 | +3.0 | smaller; worth 0.02% |
| K4L strength, age 2 / age 1 | +0.0205 / +0.0175 | 0.0078 / 0.0078 | +2.6 / +2.2 | both down |
| b source | +0.066 | 0.030 | +2.2 | down |
| X1 (birth leg) | -0.110 | 0.049 | -2.2 | up |
| k22 off-diagonal scale | +3.9 | 1.9 | +2.1 | down by about 0.004 |
| F4 theta, age 1 / age 2 | -0.025 / -0.005 | 0.013 / 0.010 | -1.9 / -0.5 | age 1 up |
| s_new / s_carry | +1.96 / +2.99 | 1.11 / 2.86 | +1.8 / +1.0 | large per network, mixed sign |
| bscale, c0, c1, iso, k31, rc, X0, X2, kappa, k4 bands | | | below 1.7 | nothing |

With the star basis held constant (4 of the 16 networks) the b source's gradient has the other sign on two networks and
bscale's on one; the consistent dials above come out similar under both rules.

Forward scan of k21n on the first 12 held-out ranks (`m_scan_inproc.py`, `cert/scan_k21n_heldout.jsonl`; S37 parameters and
physics): shift -0.02: -0.19% (se 0.16, 9 of 12 better); -0.04: +0.06% (se 0.32); -0.08: +1.75% (se 0.67). Parabola: optimum
at -0.019, worth -0.19%. The gradient had the sign right and overstated the gain three times (steep curvature).

The lead's forward scans on held-out ranks in the same hour (O/e21_fresh24_dials, O/e22_fresh24_bscale, O/e24_fresh24_shrink):
bscale up is worse (+0.48% at +0.01); b source 0.85 gives nothing once F4 is on; K4L 0.75: -0.34% (se 0.16); the learned
mean correction multiplied by 0.8: -0.84% (se 0.33), by 0.7: -0.60%.

### 9c. Run M launched; a dial-only candidate (05:33-06:10 UTC)

Launch: `TRAINER_PL=1 TRAINER_CARD=M TRAINER_BIRTH_ATTACH=1 supervisor.py run_M 2 4`, `run_M/STAR_ATTACH` = 1, and
`master.py run_M --epochs 4 --batch 12 --seed 0 --warm 4 --ema 0.98 --eval-every 12 --ckpt-every 6 --train-file cert/runB_train.txt
--val-file cert/runM_val50.txt --eval-what ema --card M --extra f4a1:0.004,f4a2:0.004,k4la1:0.01,k4la2:0.01,bsrc:0.004,rc:0.002,
k31:0.002,k21n:0.002,ds:0.01 --extra-init k4la1:-0.25,k4la2:-0.25,k21n:-0.02,ds:-0.1 --dial-lr X1:0.004`;
`announce.py run_M 0 -1.0 0.25 -0.5` (one TEAM_LOG line per held-out evaluation with the dial values; export at -1.0% with a
standard error of 0.25% or less).
- Start: the parent's arrays and dials; F4 0.6 / 0.6; K4L 0.75 / 0.75; learned mean correction x 0.9; k21n -0.048.
- `ds` is a new dial family (`otr_m.py`): the learned mean correction of layers 1-15 times (1 + shift). Natively the correction
  of a layer is linear in the literals head[l] and head_b[l], so the exporter multiplies those two: no code change, no bill.
- Trainable: 47,581 learned numbers + 10 dials + 9 physics dials = 47,600.
- Speed: 73-74 s per gradient, 454 s for the first update. A forward run takes 15-20 s per worker against 7.6-8.5 s for a
  single process on an idle box: a worker uses about one core, and the two workers and every other torch job share the
  memory traffic. More threads do not help; more workers would need 6.5-6.8 GB each at their peak.
- The rank-651 value of the card at the S37 physics equals the lead's (16.1873e-9, e22 arm S37, 3 threads).
- First update (06:05 UTC): batch-mean final-layer MSE 1.6437e-8, gradient norm 4.12 (arrays) and 2.97 (dials) before
  clipping; the extra-dial path (gradients from the per-layer leaves, Adam, shifts in the parameter files) works end to end.

Reference on the 50 held-out ranks (untrained parent on the S37 physics): final-layer MSE 1.6218e-8, all layers 6.3415e-9.

Dial-only candidates on those 50 ranks (priority evaluations through the pool before the first gradient;
`cert/runM_dial_candidates.py dialsds08 startpoint`):

| change (parent arrays otherwise) | final layer, pooled | standard error | better on | all layers |
|---|---|---|---|---|
| K4L 0.75 both ages + correction x 0.8 + k21n -0.048 | -0.980% | 0.234 | 38/50 | -0.573% |
| K4L 0.75 both ages + correction x 0.9 + k21n -0.048 (run M's start point) | -0.847% | 0.175 | 40/50 | -0.481% |

Exported: `candidates/runM_dials_ds08/candidate.py` (sha256 4de2b950e845) = the RRPI source with `_GNN_LITERAL` (head and
head_b of layers 1-15 times 0.8) and `_GN_KNOBS` (k21n -0.048) changed; the K4L strength is a lever of the engineer's builder
(`K4Lv3c0.75,0.75`); `physics_dials.json` next to it lists the values. Not run natively by this lane.

### 9d. The gain is in the dials: forward line searches along the exact dial gradient (06:00-07:25 UTC)

Method. (1) Exact gradients of every dial on a few dozen training networks (the run's own batches, or priority requests:
`m_round.py grad`), table of mean / standard error / t per dial and per layer (`m_round.py stats`, `m_run_pl_stats.py`).
(2) A step along the dials whose sign is consistent, tried at a few sizes with FORWARD runs on 16 held-out networks that are
not used for judging (`cert/ls16_ranks.txt` = positions 51-66 of the 100 held-out ranks; `m_linesearch.py read`).
(3) The chosen step is judged once on the 50 judging networks (`cert/runM_val50.txt`; `cert/runM_dial_candidates.py`).
A forward run costs a third of a gradient, and a first-order reading of a gradient overstates the gain about three times
(steep curvature: k21n, the four-dial step, k22), so the step size has to be measured, not computed.

| candidate (parent arrays; against the untrained parent on the S37 physics, 50 held-out ranks, torch, forward only) | final layer | se | better on | all layers |
|---|---|---|---|---|
| 1: K4L 0.75 / 0.75, learned mean correction x 0.8, k21n -0.048 | -0.980% | 0.234 | 38/50 | -0.573% |
| 2: candidate 1 + b source x 0.965, bscale -0.0042, X1 +0.035, F4 theta age 1 0.656 | -1.442% | 0.255 | 40/50 | -1.148% |

Candidate 2's file: `candidates/runM_dials2/candidate.py` (sha256 716f66dda0ef; written by `export_dials.py`): the RRPI
source with `_GNN_LITERAL` (head, head_b x 0.8), `_GN_KNOBS` (k21n), `_BD_J1_PHYS`, `_S19_BSCALE` and the b-source line
(`_S19_BSRC_DIV`) changed; levers of the engineer's builder: `F4v4t0.656,0.6`, `K4Lv3c0.75,0.75`. The engineer's native
session of candidate 1 gave -1.06% against the card's -0.98% (engineer 06:41Z); candidate 2 is built into the next board
file (engineer's sha 7c1a67f4, with the edge cut and the operand loads; lead 07:03Z).

Line searches on the 16 line-search networks:

| step | sizes tried: change of the pooled final-layer error against the base on the same networks | verdict |
|---|---|---|
| learned arrays: the 3,338 elements whose mean exact gradient over 36 networks is beyond 2.5 standard errors (1,776 of them in enc4_w), each moved against its gradient by the same fraction of its array's rms value (`cert/ls1`) | 0.56% of rms: +0.21% (se 0.12); 1.7%: +2.7%; 4.5%: +20.5% | nothing; optimum -0.02% |
| four dials on top of candidate 1 (`cert/ls2`): b source -0.05, bscale -0.006, X1 +0.05, F4 age 1 +0.08, times eta | eta 0.5: -0.42% (se 0.30); 1: -0.29% (se 0.42); 2: worse | eta 0.7 taken: candidate 2 |
| k22 off-diagonal scale on top of candidate 1 (3 judging ranks, one process) | -0.003: +0.65%; -0.006: +1.7% | closed: too sharp |

Exact dial gradients (d(1e8 x final-layer MSE)/d dial, mean and t over the networks):

| dial | 38 networks at run M's start point | 16 networks at candidate 2 |
|---|---|---|
| X1 | -0.176 (t -4.9) | -0.109 (t -2.0) |
| b source | +0.085 (t +3.4) | -0.010 (t -0.5) |
| bscale | +0.451 (t +3.2) | +0.109 (t +0.5) |
| F4 theta age 1 / age 2 | -0.026 (t -2.7) / -0.013 (t -2.0) | +0.007 (t +0.6) / +0.017 (t +2.1) |
| K4L strength age 1 / age 2 (at 0.75) | -0.000 / -0.002 (t -0.0, -0.4) | -0.003 / -0.001 |
| k21n (at -0.048) | +0.032 (t +0.3) | +0.137 (t +1.1) |
| correction scale (0.9, then 0.8) | +0.045 (t +2.5) | +0.030 (t +1.2) |
| edge | +0.011 (t +2.9) | +0.022 (t +3.7) |
| X2 | +0.015 (t +2.3) | +0.026 (t +3.4) |

So the four-dial step removed the gradient of the b source, bscale and F4 age 1, as intended.

Per-layer signal (exact gradients; 22 of 300 per-layer scalars beyond 3 standard errors on the first 38 networks, 0.8
expected by chance; the same signs on the 16 new networks at candidate 2 unless noted): s_carry wants DOWN at layers 1-4
(t +1.7..+2.8, then +3.7, +2.3, +2.3, +1.8) and UP at layers 9-14 (t -1.7..-5.6, then -0.7..-1.8), which cancel in the
global dial; bscale up at layers 0-5 and down at 7-10; the final layer's k4 down (t +3.1, then +2.5); k22 at layers 13-14
down (t +3.3, then +2.4, +2.5); s_new at layer 13 up on the first sample (t -4.4) but NOT on the second (+1.5).
The third line search (`cert/ls3`) tries the final-layer k4, the s_carry depth profile and four small global dials.

Run M as an optimiser: 3 updates in 68 minutes (the pool was mostly serving the evaluations above), then ended. Its
parameter files `run_M/params/p0000k.pt`, its 38 network gradients `run_M/results/g*.pt`, its unclaimed requests in
`run_M/parked/`. The learned arrays are not where the gain is on this card.

### 9e. Shrink directions, candidate 3-prelim, the candidate-3 card (07:30-08:45 UTC)

Shrink directions priced from stored gradients (`m_shrink_table.py`, no card run; lead 07:13Z). For a learned part whose output
is linear in an array X the derivative of the loss along X -> X (1 + eps) is <dLoss/dX, X>; positive = shrink. Unit:
d(1e8 x final-layer MSE)/d eps; first column 40 networks at run M's start point, second 18 other networks at candidate 2.

| group | start point | candidate 2 | reading |
|---|---|---|---|
| learned covariance edge (same number through e.output and e.cov_factor) | +0.012 (t +2.8) | +0.024 (t +3.9) | shrink |
| cstar, leg j2, all layers | +0.010 (t +2.1) | +0.023 (t +3.1) | shrink |
| cstar, leg j0 | -0.006 (t -0.8) | -0.003 (t -0.5) | nothing (legs j1, j3 have no gradient) |
| hidden-state part of the birth coefficients (A) | +0.0019 (t +2.0) | +0.0019 (t +1.2) | nothing (0.12% per unit) |
| correction scale (at 0.9, then 0.8) | +0.052 (t +2.9) | +0.040 (t +1.6) | slightly further down |

Line searches on the 16 line-search networks, on top of candidate 2 (`cert/ls3`, `cert/ls4`):

| change | pooled final-layer error against candidate 2 | better on |
|---|---|---|
| final-layer k4 x 0.96 / x 0.92 | 0.00% (se 0.08) / +0.33% (se 0.15) | 8 / 5 of 16 |
| s_carry -0.0015 at layers 1-4, +0.0015 at 9-14 / twice that | -0.11% (se 0.08) / -0.14% (se 0.16) | 10 / 8 |
| iso +0.005, X2 -0.02, edge knob -0.02, X1 +0.01 / twice that | +0.08% (se 0.07) / -0.05% (se 0.07) | 5 / 11 |
| learned covariance edge x 0.85 / x 0.70 | -0.18% (se 0.07) / -0.24% (se 0.14) | 13 / 12 |
| cstar j2 x 0.9 and correction 0.75 | -0.18% (se 0.08) | 12 |
| edge x 0.75, cstar j2 x 0.9, correction 0.75 | -0.43% (se 0.14) | 12 of 15 |

Per-layer exact gradients pooled over 58 networks: 16 of 330 per-layer scalars beyond 3.5 standard errors (0.16 expected),
all with the same sign in both samples (s_carry down at layers 1-4 and up at 14; s_new down at layer 0, up at 8; bscale up
at 0-3, down at 9-14; k22 at layer 14 down; the final layer's k4 down; c0 and k31 at layer 14 up; X1 at layer 10 up; edge at
layer 13 down). Their forward value is small: a clear sign does not mean room (the final layer's k4, t +4.2: nothing).

Candidate 3-prelim = candidate 2 + learned covariance edge x 0.75 + cstar j2 x 0.9 + correction 0.75 (was 0.8), on the 50
judging networks: -1.690% against the untrained parent (se 0.264, better on 40/50; all layers -1.303%); against candidate 2 on
the same 50: -0.252% pooled (per-network mean -0.267%, se 0.089, better on 31/50). File `candidates/runM_dials3/candidate.py`
(sha256 c9be6b965075); against the S38 parent it differs in three literals (`_GNN_LITERAL`, `_BD_COEFFICIENT_LITERAL`,
`_GN_EDGE`); builder levers as S38.

The candidate-3 card (`otr_m.py`): a request with `f4c: 1` puts the physicist's `f4b.py` (mode 'mix', metric 'birth', star
anchor) in place of the F4 block; the rule strength per pair age and layer is the F4 theta leaf, the exact S1 / S2 strengths
per age are leaves `x1a1 x1a2 x2a1 x2a2` (base 0); `e2` is a per-layer leaf for the lead's second-order skew term of the mean
formula (base 0). Certificate on rank 651 (`m_certify_f4c.py`, `cert/m_certify_f4c/`, 3 threads): with rule 0.35, exact 0.5,
K4L 0.75 the card equals the physicist's own call plus the lead's K4L hooks on all 16 layers to the last bit; at the fitted
literals (0.378, 0.48, 0.71) the two differ by 3e-5 of the MSE (rounding in the order of the K4L strength, amplified by the
card's precision noise); with the exact strengths at 0 it equals the F4 card to 1.5e-4 of a layer's MSE; the e2 leaf equals
the lead's hook to the last bit (+0.755% at strength 1). The restarted workers reproduce the certificate's values on rank 651.

`m_mix.py`: the physicist's mixture method in this lane's pool: one forward run per single change with the final-layer
predictions kept; the prediction is close to linear in small dial changes, so the pooled error is a quadratic in the weights
of the changes and its minimum comes from one least-squares solve (leave-one-network-out for an honest gain).

QUIET WINDOW 07:25-07:57 UTC: the pool ran 5 min 50 s into it (the watcher's pattern used an alternation BSD grep does not
accept), and two forwards ran for 21 s at 07:36:43 (requests queued in the window were picked up by the idle workers before
the watcher moved them aside). `quiet_watch.sh` now counts only [lead] lines that begin with the announcement; nothing is
queued inside a window any more.

### 9f. The joint solve on the exact-term card: candidate 3 (08:42-10:20 UTC)

`m_mix.py queue / solve` (label mix3b, `cert/mix3/spec_v2.json`): base = candidate 3-prelim on the exact-term card (rule
strength 0.656 / 0.6, exact strengths 0); 12 single changes, one forward run each on the 16 line-search ranks with the final
predictions kept (208 runs, 53 minutes under load); pooled least squares with a relative ridge of 0.02.

| single change (size) | alone | prediction change / error rms | weight | leave-one-out spread |
|---|---|---|---|---|
| rule strength age 1 -0.2 | +0.42% | 0.054 | -0.11 | 0.64 |
| rule strength age 2 -0.2 | +0.02% | 0.041 | +0.06 | 0.51 |
| exact S1 age 1 +0.48 | +0.18% | 0.066 | +0.24 | 0.37 |
| exact S1 age 2 +0.48 | +0.34% | 0.049 | -0.23 | 0.30 |
| exact S2 age 1 +0.48 | -0.32% | 0.056 | +0.85 | 0.29 |
| exact S2 age 2 +0.48 | -0.04% | 0.040 | +0.40 | 0.63 |
| K4L -0.1 | +0.16% | 0.023 | -1.19 | 0.62 |
| correction scale -0.1 | +0.28% | 0.042 | -0.27 | 0.35 |
| e2 +0.5 at layers 1-7 | +0.11% | 0.020 | -0.79 | 1.13 |
| e2 +0.5 at layers 8-15 | +0.15% | 0.076 | +0.35 | 0.24 |
| edge x 0.8 more | +0.01% | 0.022 | +0.39 | 0.73 |
| cstar j2 x 0.85 more | -0.08% | 0.023 | +0.84 | 0.54 |

The single changes moved the prediction by 2-8% of the error's rms, the size of the card's per-network precision noise, so
most weights are noise: the model promised -0.62% and its leave-one-network-out estimate was +0.12%. The optimum run for real:

| optimum (run on the 50 judging ranks, which the solve never saw) | against the untrained parent | against candidate 2 | against 3-prelim | all layers against the parent |
|---|---|---|---|---|
| all 12 changes (`cert/mix3/mix3b_optimum.pt`) | -2.300% (se 0.279, 43/50) | -0.870% (se 0.187, 38/50) | -0.620% (se 0.144, 38/50) | -1.489% |
| five changes: exact S2 at both ages, K4L, cstar j2, correction (`mix3b_optimum_min5.pt`; model -0.47%, leave-one-out -0.20%) | -2.134% (se 0.257, 45/50) | -0.702% (se 0.152, 35/50) | -0.451% (se 0.111, 37/50) | -1.704% |

On the 16 line-search ranks the 12-change optimum's real run gave -0.58% (se 0.25) against its base (model -0.62%).
CANDIDATE 3 = the five-change optimum: exact S2 strength 0.444 (age 1) / 0.171 (age 2), exact S1 0, rule 0.656 / 0.6, K4L
0.860, correction x 0.784, cstar j2 x 0.791, edge x 0.75, the rest as candidate 2. File
`candidates/runM_cand3_five/candidate.py` (sha256 a4515f39c626) with `physics_dials.json` (the strengths are builder levers).
It needs no port of the second-order skew term; the 12-change optimum does (e2 0.18 at layers 8-15) and is 0.17% (se 0.08)
better on the final layer, 0.22% worse over all layers.
Both raise K4L from 0.75 to 0.86 (the lead warns that kurtosis terms can hurt at depth; the 50 held-out ranks are the check).

Round 2 (label mix4, `cert/mix4/spec.json`): 13 single changes three to four times larger around the 12-change optimum, K4L
and e2 split into layers 1-7 and 8-15; `m_mix_driver.sh` waits, solves, runs the optimum on the line-search and judging ranks
and writes the verdict (`m_mix_verdict.py`) to TEAM_LOG.

Pool notes: the supervisor's shed rule took both workers out from 09:53 to 10:03 UTC (system swap grew 227 MB when four other
torch jobs started); beside those jobs a forward run takes 25-30 s per worker.

### 9g. The lead's carrier module in the pool's card (10:20-10:50 UTC)

`otr_m.py`: a request field `ncar` (a dict; `{}` = the defaults) installs the lead's `O/ncar.py` last, after K4L and F4 or the
exact-term module. The install call's arguments are in one dict (`NCAR_ARGS`: K 16, modes 1..K, ca 0.5, cb 0.25, layers 2-14,
conv 1, law 2); the three strengths are leaves read when the module multiplies with them (`nk31` per layer, base 1; `nct`
transport, base 0.9; `ncs` sources, base 1). The module's hooks are composed, not chained: this card's own `pair_in` runs first
with gradients, then the module's part runs under no_grad on detached inputs and only its remainder31 plane is taken; so in a
gradient run the plane is a constant. Every result records the arguments used (`ncar`).

Certificate, rank 651, 4 threads, through the restarted workers, F4 0.6 + K4L 0.75 (the lead's e37 arms in brackets): no
carrier 16.0078e-9 (16.0081); K 16 modes 1-16 15.0285e-9 (15.0285); K 8 modes 1-8 15.3694e-9 (15.3694); K 16 modes 2-16
15.0572e-9 (15.0575).

Judgement on the 50 held-out ranks (torch, forward only, exact-term card; base = candidate 3, the five-change optimum):

| card | final-layer MSE | against candidate 3 | against candidate 2 (S38 dials) | against the untrained parent | all layers against candidate 3 |
|---|---|---|---|---|---|
| candidate 3 | 1.5872e-8 | | -0.702% | -2.134% | |
| candidate 3 + carrier, K 16, modes 1-16, strength 1 | 1.5546e-8 | -2.057% (se 0.256, 42/50) | -2.745% (se 0.251, 48/50) | -4.147% (se 0.352, 47/50) | -1.925% |

Per network against candidate 3: quartiles -3.5 / -2.1 / -0.7%, best -6.6%, worst +1.8%, standard deviation 1.8%; per layer
(pooled): 0 at layers 0-2, -0.5% at 3, -1.0..-1.7% at 4-7, -1.9..-2.3% at 8-15. Per network the effect depends on the dials
underneath (the first five held-out ranks: -2.0, +0.5, -0.3, -5.7, -1.3% here; -6.1, -1.7, -4.6, +0.8, +0.2% in the lead's
screen on the S37 dials). A forward with the carrier: 38.5 s in the pool against 33.2 s without.

K = 8 (modes 1-8, otherwise the same) on the same 50: final-layer MSE 1.5732e-8; against candidate 3 -0.885% (se 0.202, 38/50;
all layers -1.008%); against candidate 2 -1.581% (se 0.214, 43/50); against the untrained parent -3.000% (se 0.313, 47/50).
K = 16 against K = 8: -1.182% (se 0.164, better on 45/50). Seconds per forward in the pool: 51 at K = 8 (measured under heavier
load), 38.5 at K = 16.

The sketched carrier (`O/ncar4.py`, lead 11:16Z): a request with `ncar = {"module": "ncar4", ...}` installs it instead of the
vector form (`NCAR4_ARGS`: r 32, k 24, K 8, conv 1, k21 on, lean, layers 2-14, start 0, two passes; only the plane strength
`nk31` is a leaf). To be judged on the 50: (1) as called, (2) one pass from layer 6, (3) r 64 / k 40 only if (1) beats the
vector form clearly. `cert/car/after_window.sh` restarts the workers at the window's close, certifies on network 35 against the
lead's 13.0430e-9 / 12.6729e-9, queues the judgements and writes one TEAM_LOG line per result (`cert/car/line.py`).
The carrier-strength solve (label car1, `cert/mix5/spec.json`) is parked in `run_M/parked_car1/` with 28 of 96 runs done.

### 9h. The carrier's forms on the 50 held-out ranks (12:00-12:50 UTC)

The card takes carrier modules through the request field `ncar` (one dict or a list; `otr_m.py`, table `MODULES`): `ncar`
(vector form), `ncar4` (sketched state), `ncar5` (structured low-rank state without probes), `bank` (the physicist's bank
channel, `O/physicist/ncar_bank_frozen_s2s.py`, strength leaf `nkbk`), `ncarry` (the physicist's full matrix). Each module's
hooks are composed after this card's own and run without gradient. Certificates on rank 651 / network 35 against the lead's
harness (4 threads, F4 0.6 + K4L 0.75): vector K 16 15.0285e-9 (15.0285), K 8 15.3694 (15.3694); sketched 12.6733 (12.6729,
network 35); bank 15.2659 (15.2655); ncar5 15.5087 (15.5085); both 14.7998 (14.7999); full matrix with the old row term
13.4924 against the lead's 13.5970 (NOT reproduced: `ncarry.py` changed at 11:59 UTC, after the lead's run).

All on top of candidate 3 (1.5872e-8), torch, forward only, exact-term card, strength 1; `cert/car/line.py <label> <text>`:

| carrier form | ranks | final-layer MSE | against candidate 3 | better on | against the vector form K 16 | all layers against candidate 3 |
|---|---|---|---|---|---|---|
| vector form, K 16, modes 1-16 | 50 | 1.5546e-8 | -2.057% (se 0.256) | 42/50 | | -1.925% |
| vector form, K 8 | 50 | 1.5732e-8 | -0.885% (se 0.202) | 38/50 | +1.20% | -1.008% |
| sketched rank 32, 24 probes, two passes, from layer 0 | 50 | 1.5807e-8 | -0.410% (se 0.285) | 30/50 | +1.682% (se 0.362) | -0.589% |
| sketched, ONE pass, from layer 6 | 50 | 1.6586e-8 | +4.497% (se 0.459) | 4/50 | +6.692% (se 0.551) | +2.442% |
| bank channel alone (second weight per bank column) | 50 | 1.5499e-8 | -2.354% (se 0.254) | 47/50 | -0.303% (se 0.287) | -2.669% |

Bank channel per network: quartiles -3.4 / -2.2 / -0.9%, best -7.6%, worst +1.7%; per layer -1.0% at layer 3 and -2.5..-3.0%
from layer 5 on. Its first six ranks read -5.4, -3.2, -2.8, -3.1, -4.3, -1.3% against the physicist's -5.7, -2.8, -2.6, -3.3,
-4.4, -1.0% on the S38 dials: this piece does not depend on the dials underneath (the vector form does), and those six ranks
are above the 50's average. Correlation across networks between the bank channel's and the vector form's effect: 0.39.
Single ranks overstate the whole line: rank 651 reads two to three times the pooled value for every form.

`cert/car/after_window.sh` and `cert/car/after_c3s4b_v2.sh` are the scripts that ran these without anyone awake (restart of
the workers for new code, certificate first, a parked queue and a TEAM_LOG line if a certificate fails).

### 9i. The carriers that live inside the bank read: the fold and the pair-carried law (12:50 UTC on)

Pool order changed on the lead's 12:21Z / 12:44Z lines: bank + ncar5 together (16 of 50 done: -3.64% against candidate 3,
se 0.66; -1.22% against the bank channel alone, se 0.41), ncar5 alone and round 2 of the dial solve are parked as superseded
(`run_M/parked_superseded_1250/`, `run_M/parked_mix4/`), the full-matrix ceiling is parked for last (`run_M/parked_c3yfull/`).

New in the card (`otr_m.py`; version before: `cert/otr_m_v8_before_fold.py`):
- `{'module': 'fold'}`: the physicist's bank channel folded inside the bank read (`O/physicist/ncar_fold_frozen_s2w.py`,
  level 'read', layers 1-14, forms 1, diagonal regression with three regressors, factor 1.41 on the removed multiple of C).
- `{'module': 'ncar6'}`: the lead's pair-carried lean law (`O/ncar6.py`, level 'read', forms 1, dfit 2 = per-layer literal x
  diagonal regression, ca 0.75, arest False). It reads the layer's selected hubs and their two hub entries from the side table.
- Both wrap the card's bank read and chain the 'mu' / 'cpre' hooks; nothing of this lane sits on those. Their dials are leaves
  around the request's values: `nfs` (strength), `nfd` (factor on the removed multiple of C), and for ncar6 `n6b`, `n6r`, `n6d`
  (births, rows, diag). A request may give any install argument (`forms`, `dscale`, ...).
- `'ss': 0` in a request: the engineer's star by row sums (`O/engineer/a33_starsum/star_ss_torch.py`) in place of the card's
  rank-32 star = the S39 card (`otr_engine.STAR_SS`; versions before: `cert/otr_engine_before_star_ss.py`, `cert/worker_before_star_ss.py`).

Certificates (torch, forward only, 4 threads, strength 1; final-layer MSE in 1e-9, mine / theirs):

| what | network | base | with the carrier |
|---|---|---|---|
| fold, one form (physicist, S38 physics) | 651 | 15.6230 / 15.6222 | 14.9195 / 14.9178 (-4.50% / -4.51%) |
| fold, three forms | 651 | | 15.0027 / 15.0011 (-3.97% / -3.98%) |
| fold, one form | 35 | 13.0225 / 13.0215 | 12.5246 / 12.5256 (-3.82% / -3.81%) |
| fold, three forms | 35 | | 12.4931 / 12.4926 (-4.07% / -4.06%) |
| ncar6 (lead's e51 rows, S37 physics) | 378 | 14.0280 / 14.0283 | 13.3041 / 13.3041 (-5.16%) |
| ncar6 | 35 | 13.0430 / 13.0429 | 12.3643 / 12.3643 (-5.20%) |

Parity of the S39 star with the native rows (`cert/fold/ss_parity.py`; ranks 651 / 378, final layer): change of the prediction
from the star by row sums on candidate 2's dials, torch against native: rms 1.17e-5 / 1.11e-5 against 1.25e-5 / 1.22e-5,
cosine 0.934 / 0.931, left over 4.5e-6 / 4.4e-6 rms (the level at which two native builds differ); the whole step S38 -> S39:
cosine 0.957 / 0.954, left over 4.6e-6 / 4.9e-6. The star by row sums alone moves the final-layer MSE by -0.25% / +1.17% in
torch and +0.04% / +1.46% natively. Torch card minus native file: 1.7e-5 / 1.8e-5 rms (14% of the error's rms).

Judgements on top of candidate 3 (torch, forward only, exact-term card WITHOUT the S39 star, because every reference on the 50
is on that card; `cert/fold/queue_judgements.py`, `cert/fold/driver.sh`, `cert/fold/line.py`):

| variant | ranks | final-layer MSE | against candidate 3 | better on | against the bank channel through a separate plane |
|---|---|---|---|---|---|
| fold, one form removed | first 25 | 1.5314e-8 | -2.469% (se 0.407) | 23/25 | +0.319% (se 0.120) |
| ncar6, one form removed | first 25 | 1.4748e-8 | -6.072% (se 0.758) | 24/25 | -3.387% (se 0.714) |
| ncar6, three forms removed (forms 3, dfit 2) | first 25 | 1.4668e-8 | -6.583% (se 0.789) | 25/25 | -3.912% (se 0.726) |
| fold, three forms removed | first 25 | 1.5330e-8 | -2.365% (se 0.386) | 23/25 | +0.427% (se 0.108) |
| **ncar6, one form removed** | **50** | **1.4951e-8** | **-5.804% (se 0.499)** | **47/50** | **-3.534% (se 0.468)** |
| ncar6 with a constant regressor instead of the table, three forms (physicist's audit copy, dfit 6) | first 25 | 1.4692e-8 | -6.433% (se 0.771) | 25/25 | -3.758% (se 0.740) |
| **ncar6, three forms removed (table)** | **50** | **1.4882e-8** | **-6.237% (se 0.508)** | **50/50** | **-3.977% (se 0.474)** |
| law (constant regressor, three forms) + k31l (lead's fourth-chaos plane of the young pairs, ages 1-2, strength 1) | first 25 | 1.4417e-8 | -8.180% (se 0.839) | 25/25 | -5.555% (se 0.737) |

k31l against the law alone (constant regressor, three forms), first 25 ranks: -1.867% (se 0.388, better on 21/25; all layers
-1.73%). Three forms against one form on all 50: -0.459% (se 0.135, better on 34/50). Certificates of k31l on 651 (S37
physics, mine / lead's, 1e-9): law alone 13.9320 / 13.9322; with k31l in the read 13.6413 / 13.6415; through remainder31
13.6592 / 13.6598; three instances (age limits 1 / 2 / 3 with strengths 0 / 1 / 0, the form the dial solve uses) 13.6413.

Paired on the first 25 ranks: ncar6 three forms against one form -0.544% (se 0.207, better on 18/25); fold three forms against
one form +0.107% (se 0.073, better on 9/25). The law per network (one form, first 25): quartiles -7.8 / -5.3 / -3.4%, best
-13.8% (651), worst +0.1% (473); per layer -1.5% at layer 2, -4.0% at layer 3, -6 to -7.7% at layers 4-15; all layers on the
50: -6.29%. On the six ranks shared with the lead's e51 table the law gives about 1.2 points more on candidate 3 than on the
S37 physics with default dials. Against candidate 2 (S38 dials) on the 50: -6.47%; against the untrained parent on the S37
physics: -7.81% (50 of 50).

Constant regressor against the table, three forms both, first 25 ranks: +0.161% (se 0.162, better on 11/25): the same within
noise, so the 14-literal table (fitted on 651 and 35) can be dropped without loss; the physicist's +0.9 points on network 35
does not hold pooled.

More certificates (S37 physics, final-layer MSE in 1e-9, mine / theirs): the lead's three on 651: plane through remainder31
with four exact forms 13.6228 / 13.6233, fold with four exact forms 13.7920 / 13.7924, fold with one exact form 13.9866 /
13.9865. The physicist's audit copy (`n6a` = `O/physicist/ncar6_audit_frozen_s3d.py`): table on 378 13.3041 / 13.3041; constant
regressor one form 12.2424 / 12.2421 (35) and 13.9724 / 13.9724 (651); three forms 13.9320 / 13.9322 (651); K22 companion at
strength 1 on 35 12.5741 / 12.5741 and at 0.5 on 651 13.8579 / 13.8581 (a module's change of the K22 plane in 'pair_in' is added
by the card's composition since 13:56Z). The pool uses a copy of the lead's file as of 13:39Z (`frozen/ncar6_t1339.py`); after
the reload it gives 13.3041e-9 on 378 as before.

Mistake on record (13:55Z): requests for the new module name were queued before the workers were reloaded; the old workers
answered 17 of them with KeyError within a second, a stage script wrote two false failure lines to TEAM_LOG (corrected there);
error and claim files moved to `run_M/errors_n6a_1356/`, requests queued again. Also: `cert/fold/queue.py` shadowed the standard
library's `queue` for scripts run from that folder and is now `cert/fold/queue_judgements.py`.

Dial solves on top of the law (S39 card = request field `'ss': 0`; `m_mix.py` takes a stored candidate as its base through
`"base": {"file": ...}`; `cert/fold/dial_driver.sh`, `cert/fold/dial_verdict.py`, `cert/fold/after_d6.sh`):
- `cert/fold/spec_ncar6.json`, label `c3q4zd6`: the law's own dials (strength in three layer bands, scale on RHO1, births,
  rows, diag) plus the card's c0 and k31 knob, three forms removed; 16 line-search ranks; optimum and default dials then on
  the first 25 judging ranks.
- VERDICT of `c3q4zd6` (15:13Z): model -1.80% on the 16 ranks (leave-one-out -1.30%), real run of the optimum -1.43% (se
  0.31, 14/16); JUDGED on the first 25 judging ranks against the same card and law at default dials: **-1.479% (se 0.283,
  better on 20/25; all layers -0.765%)**. Optimum (`cert/fold/c3q4zd6_optimum.pt`): fold strength 1.83 / 1.40 / 1.18 at
  layers 1-4 / 5-9 / 10-14; scale on the removed multiple of C 1.27; births 1.14, rows 0.70, diag 0.79; c0 +0.088; k31 knob
  +0.046. Single arms on the 16 ranks: strength +0.5 at layers 1-4 -0.21% (se 0.13), 5-9 +0.45% (0.33), 10-14 +2.05% (0.21);
  RHO x 0.7 +2.98% (0.43); births off +2.63% (0.35); rows halved +0.39% (0.48); diag off +1.40% (0.46); c0 -0.1 +0.86%
  (0.22); k31 knob -0.2 +1.43% (0.30).
- second solve, label `c3q4zj7`, `cert/fold/spec_d7.json` (queued 15:15Z): base = that optimum, law with the constant
  regressor (`n6c`), k31l as THREE instances with age limits 1 / 2 / 3 and leaves nkl1 / nkl2 / nkl3 (arms that move one
  cohort age only: age 1 = nkl1 +0.5; age 2 = nkl2 +0.5, nkl1 -0.5; age 3 = nkl3 +1, nkl2 -1), plus the law's nine dials again.
- chained behind it (`cert/law_ls/chain.sh`): the learned-array step test on the card with the law (36 exact gradients,
  `cert/law_ls/linesearch_law.py`) and round 2 of the card's own dials (label `c3q4zr2`).

Gradient runs with carrier modules (`otr_m.py`, comment REPLAY; version before: `cert/otr_m_v12_before_replay.py`): a step that
the checkpoint runs again takes the modules' contributions from a record of the forward pass. Certificates on the 16
line-search ranks (S39 card + law, table, three forms): forward error of the gradient run = forward-only run on all 16; the
replayed K21 plane equals the forward pass (asserted); c0 -0.1: first-order +0.43%, real +0.86% (se 0.22), per-network
correlation 1.00; k31 knob -0.2: +0.59% against +1.43% (se 0.30), correlation 0.99. 56 s per gradient run.
Since 15:34Z the value of the bank read in a gradient run is the modules' read bit for bit (`record + (plain - plain.detach())`:
the gradient is the card's own read's): with two and with four modules installed the gradient run's forward error equals the
forward-only run exactly (law + k31l on rank 3, law + three k31l instances on rank 22); the first form (`plain + (modules -
plain)`) differed by 1e-4 of the error there through float32 rounding. Peak memory of a gradient run: 7.5 GB per worker (as
before the law; the supervisor's gate budgets it).

### 9j. Second dial solve, the age-3 question, the tier scales (15:15-17:30 UTC)

Second joint solve (label `c3q4zj7`, `cert/fold/spec_d7.json`; S39 card, law with the constant regressor and three forms, k31l as
three instances with age limits 1 / 2 / 3): model -0.57% on the 16 line-search ranks (leave-one-out +0.15%), real run -0.50% (se
0.24, 13/16). Judged on the first 25 judging ranks against the same card with the first dials and k31l on cohorts of ages 1-2 at
strength 1 (= the S41 stack):

| second-solve dials with k31l at | against the S41 stack | better on |
|---|---|---|
| ages 1-2 at 1, no age 3 (`cert/fold/c3q4zj7_ageA.pt`; exported `candidates/law_dials_d7_ages12/`) | -0.840% (se 0.194) | 21/25 |
| ages 1-2 at 1.02 / 0.77, no age 3 (`c3q4zj7_ageB.pt`) | -0.762% (se 0.198) | 21/25 |
| ages 1 / 2 / 3 at 1.02 / 0.77 / 0.72 = the full optimum (`c3q4zj7_optimum.pt`; exported `candidates/law_dials_d7/`) | -1.027% (se 0.206) | 19/25 |

Paired: the whole k31l change -0.188% (se 0.124); the age-3 term given 1.02 / 0.77: -0.267% (se 0.121); age 2 lowered alone
+0.079% (se 0.053). The verdict line's nkl1 / nkl2 / nkl3 are INSTANCE shifts (age limits, bases 0 / 1 / 0); strengths by
cohort age are their sums (`export_dials.py` writes the sums). Optimum values: fold strength 1.850 / 1.207 / 1.167 at layers
1-4 / 5-9 / 10-14, factor 1.307, births 1.305, rows 0.854, diag 0.905, c0 0.6954, k31 knob -0.1451 (candidate.py sha
c2b7f6839646 in both exports).

Tier scales at the bank read only (`otr_m.py`, leaves `rso` old tier, `rsy` young tier; zero shift bit-identical): exact
gradients on six networks: young tier +1.04% per +0.01 (t +2.3; positive at layers 1-7, mixed at 8-14), old tier no common sign
(-0.06%, t -0.7). Finite-difference control on four line-search ranks: correlation of predicted and real change 1.00 (old) and
0.99 (young); the young tier at 1.01 costs +2.68% (first order +1.38%; curvature 26,000% per unit squared): optimum near
-0.005, about -0.4%. Small joint solve `c3q4zq1` (`cert/fold/spec_tier.json`) and round 2 of the card's dials `c3q4zr2`
(`cert/fold/spec_round2.json`, nine arms) are queued on the base `c3q4zj7_ageA.pt`.

Pool notes: gradient runs with four modules installed take 2-3 minutes and 5-7 GB on the loaded machine; the supervisor shed
both workers at 16:57Z (swap +175 MiB when two gradient runs started together); since 17:08Z one worker (the gate keeps the
second out at 10-13 GB free + inactive). 18 of the 36 gradient runs for the learned-array test are parked
(`run_M/parked_lawg/`; `cert/law_ls/step_test.sh <base>` runs the test once they are back in the queue).

### 9k. Tier scales judged; new head inputs built (17:30-18:00 UTC)

Tier scales (`c3q4zq1`): single arms on the 16 line-search ranks: young tier -0.5% at layers 1-7: -0.209% (se 0.245, all layers
+0.39%); at layers 8-14: +0.140% (se 0.101); old tier +1%: +0.031% (se 0.072). Joint optimum: model -0.27% (leave-one-out
-0.02%), real run -0.26% (se 0.25, 9/16); JUDGED on the 25 judging ranks -0.226% (se 0.143, better on 13/25; all layers
+0.457%). Not a lever: the gradient is real (finite-difference correlation 0.99-1.00) but the curvature leaves nothing.

Round 2 of the card's dials (`c3q4zr2`, nine arms: ds, edge, K4L, exact S2 per age, F4 rule, k21n, b source, bscale) runs on
the base `c3q4zj7_ageA.pt`; `cert/fold/after_r2_export.sh` exports it (`candidates/law_dials_r2/`) only if its judged gain is
beyond two standard errors.

New head inputs (lead 17:52Z; `otr_m.py` comment HEAD INPUTS; request field `hf`, weights `hu`; law module `n6x` =
`frozen/ncar6_t1402x.py`, the lead's file with its state exposed): the learned mean correction is 30 standardised features +
96 message columns -> enc0 (128 x 126, silu) -> enc2 -> enc4 (64, the unit's state, which also feeds the next layer's
messages) -> per-layer head. A new column enters as a per-unit bias of enc0 (`enc0_b + z_i u'`, u = 128 weights at zero):
predictions unchanged bit for bit, gradient of u exact. Six columns (HEAD_COLUMNS). Certificates on rank 22: exposed-state copy
= unchanged copy; features on with zero weights = off; gradient run: same forward value, same array gradients; finite
difference along minus the 768 weights' gradient: step 0.02 -0.526% real against -0.528% first order, step 0.05 -1.220%
against -1.321%. The t table over 36 training networks is produced by `cert/law_ls/after_round2_v2.sh` (18 owed gradient runs
+ 18 repeats with the head inputs) and `cert/law_ls/head_input_table.py`.

### 9l. The law matrix's diagonal as a new head input: the 128 weights solved (18:20-19:25 UTC)

Round 2 of the card's dials (`c3q4zr2`): model -0.56% (leave-one-out -0.07%), real -0.36% (se 0.26); judged -0.403% (se 0.216,
17/25): not beyond two standard errors, not exported; the second dials stand.

Head-input test over 36 training networks (zero weights; `cert/law_ls/head_input_table.py`): column 0, the unit-specific
part of the law matrix's diagonal: 105 of 128 weights beyond 3 standard errors, mean t squared 54.9, split-half cosine 0.98,
first-order -0.78% per step of 0.05; column 5 (fit over B_ii) the same signal; the fold terms' row norms and row sums weak
(t squared 1.3-3.6); existing columns 1.4-2.9 except the trained `pre k4` column (64 beyond 3, t squared 11.3, halves 0.93).

The column in the NATIVE convention (`otr_m.py` column 6, `diag_resid_rows`; engineer: the native fold holds the diagonal on
the read's response rows only): rows = the layer's response rows (1,024 at layers 1-2 falling to about 600-700 at layers
12-14); B_ii = sum_r (w_r pp_ir^2 + x_r pp_ir pp_i,partner(r)); fit over those rows on (tr C v_i, 1, mu_i^2, 2 mu_i (C mu)_i);
x_i = (B_ii - fit_i) / (tr C v_i); z_i = clip((x_i - mean) / sd, -8, 8) over the rows (sd with n - 1); 0 for every other unit;
layers 1-14. It enters enc0 as a 31st feature column (index 30, centre 0, scale 1).

Solve of its 128 weights (`cert/head/solve.py`, `cert/head/steps.sh`; base = second dials with k31l as in S41,
`cert/fold/c3q4zj7_ageA.pt`; request `hf: 2`):

| step | direction (36 training networks) | line search on the 16 line-search ranks | judged on the first 25 judging ranks |
|---|---|---|---|
| 1 | at zero weights: 105/128 beyond 3 se, t squared 54.9, first order -15.6% per unit | 0.1: -1.154% (16/16); 0.25: -1.740%; 0.5: +0.031%; minimum 0.248 (curvature 28.3) | against the card without the column **-1.787% (se 0.286, 23/25; all layers -2.028%)**; against the first dials without it (S41 arithmetic in torch) -2.612% (se 0.344) |
| 2 | at step 1's weights: 43/128, t squared 9.1, halves 0.82, first order -4.25% per unit | 0.038: -0.130% (se 0.036); 0.075: -0.231%; 0.188: -0.389%; minimum 0.21 (curvature 8.95), against step 1 | against the card without the column -2.144% (se 0.304, 24/25; all layers -2.27%); against step 1 -0.364% (se 0.152, 18/25) |

The all-units version of the column (column 0) gives -1.137% at step 0.1 against -1.154% for the response-rows version: the
native convention loses nothing. Hand-over for the engineer: `candidates/head_col_s1/` (u_128.npy sha 6312b3df6dbd, enc0_w_127.npy
with the column at index 30, column.json, torch predictions of all 16 layers on 651 / 378 with the column off and on).

Second batch of head inputs (lead 18:42Z; columns 7-11, modules `n6y` / `k31ly` = `frozen/ncar6_t1402y.py`,
`frozen/k31l_t1508y.py`, certified bit-identical): the two parts of the diagonal, the diagonal of the young pairs' plane, the
previous layer's sources rv and d o Pv pushed through W. Their zero-weight gradients ride on the solve's gradient runs
(`cert/head/batch2_table.py`); `cert/head/after_solve.sh` writes the table and runs the learned-array step test with the column
in place.

Judged on the S41 ARITHMETIC in torch (S39 card, law with the TABLE and three forms, FIRST dials `c3q4zd6_optimum.pt`, k31l
ages 1-2; ids `a_0hS41_base / _u1 / _u2`), 25 judging ranks, column on against off: step 1's weights -1.920% (se 0.279, 24/25;
all layers -2.129%); step 2's -2.180% (se 0.305, 24/25; all layers -2.310%); step 2 against step 1 -0.265% (se 0.151, 14/25).
The S41 arithmetic without the column is level with the second dials + constant regressor (-0.219%, se 0.153) and 1.06% (se
0.19) better than the first dials on the constant regressor (the first dials belong to the table form). Native (engineer, ten
fresh networks, S41 file + the column with step 1's weights): -1.55% pooled, 8 of 10 lower, bill -0.03%: S42.

In the card since 19:28Z (`otr_m.py` v18 backup `cert/otr_m_v18_before_third_batch.py`): columns 7-16 (second and third batch:
diagonal parts, plane diagonal, pushed sources rv and d o Pv, k3 of young / old / star / b-form, the K4L local quartic's value);
`cert/head/solve.py` takes several columns in one direction (`direction <labels> <col,col,...> <out>`); the driver
`cert/head/steps.sh` sizes its steps from the previous step's curvature; `cert/head/after_solve.sh` writes the zero-weight table
of both batches (`cert/head/batch2_table.py`) and runs the learned-array step test with the column in place (`cert/law_ls2/`).

### 9m. S42 fired; batches 2-3 of head inputs; the joint third step; batch 4 readouts (19:35-20:15 UTC)

S42 (#333607, fired 19:49Z) = S41 + the diagonal head column at step 2's weights (`cert/head/v2r_weights.json`, row 6): native
on ten fresh networks -2.15% (10 of 10), torch -2.18% on the 25 judging ranks on the S41 arithmetic. From 19:43Z the head-input
work runs on the S42 ARITHMETIC in torch (`HEAD_CARD=s42` for `cert/head/solve.py`: law with the table, `dfit 2`; parameters =
first dials `cert/fold/c3q4zd6_optimum.pt`; stored: 25 judging ranks `a_0hS41_u2`, 16 line-search ranks `a_0hT2bj`, 36 exact
gradients `a_0hT2g`).

Zero-weight table at S42's point (36 training networks; `cert/head/batch2_table.py 0hT2g`; weights of 128 beyond 3 se / mean t
squared / split-half cosine / first order per step of 0.05, noise out):

| column | beyond 3 | mean t squared | halves | first order |
|---|---|---|---|---|
| 6 diagonal residual, at its solved weights | 17 | 3.86 | +0.69 | -0.162% |
| 7 carried-column part of the diagonal | 3 | 1.83 | +0.57 | -0.075% |
| 8 pair cross part of the diagonal | 3 | 1.81 | +0.56 | -0.072% |
| 9 diagonal of the young pairs' fourth-chaos plane over v^2 | 74 | 13.38 | +0.93 | -0.300% |
| 10 / 11 previous layer's rv, d o Pv pushed through W | 0 / 0 | 0.87 / 0.44 | -0.38 / -0.54 | 0 |
| 12 / 13 / 14 / 15 k3 of young columns, old tier, star, b-form | 0 / 2 / 0 / 3 | 0.85 / 2.01 / 2.12 / 1.60 | | 0 to -0.10% |
| 16 the K4L local quartic's value over v^2 | 29 | 5.51 | +0.82 | -0.174% |
| existing pre k4 column (trained) | 24 | 4.49 | +0.79 | -0.297% |

Joint step over columns 6, 9, 16 (384 weights, one direction; first order -7.65% per unit step): 16 line-search ranks against
S42's weights: 0.1: -0.226% (se 0.231, 9/16); 0.25: +1.267%; 0.5: +7.46%; minimum -0.20% at 0.062 (curvature 40). Not beyond
its noise; not judged, not handed over. The mean gradients of columns 9 and 16 have cosine 0.92 / 0.89-0.94 with column 6's
over the 128 first-layer units: the same move, already made. Item (c) (left-factor row sizes) dropped (0.16% of bill for a
stand-in of weak columns).

Batch 4, readouts with zero weights (`otr_m.py` comment READOUTS; request `rg`, weights `rgw`; version before:
`cert/otr_m_v19_before_readouts.py`): `row` = row gain of the bank read, r_i = q_i . (B0 + (l - 3) B1), 52 inputs standardised
over the response rows, layers 2-14, applied to the whole returned plane; `b` = carried b, 64 weights per layer; `rad` =
radial, 64 per layer. Certified on rank 3 (S42 arithmetic): forward and gradient run identical to the stored value
(1.585407264e-8), array gradients unchanged. Gradient runs `a_0hT2rg` (36 training networks), table `cert/head/readout_table.py`,
solver `cert/head/solve_rg.py`.

Learned-array step test with the column in place (`cert/head/arrays_test.sh`, `cert/law_ls3/`): direction from the same 36
gradients: 5,750 elements beyond 2.5 se (enc4_w 1,772, enc2_w 1,572, enc0_w 1,172, msg_w 162); shares of the first-order gain:
enc4_w 44%, enc0_w 20%, enc2_w 16%, edge feature scale 14%.

### 9n. Final week (from 2026-10-10 02:00 UTC): batch 4 judged, final-layer columns, the S43 joint solve

S42 #333607 graded 1.667e-9 (third; -2.6% against S41). The learned-array step test with the column in place (S42 arithmetic,
16 line-search ranks, `cert/law_ls3/`): eta 0.0044: -0.160% (se 0.091, 11/16); eta 0.0132: +5.85%; parabola minimum at eta
0.0025: -0.35%.

Batch 4 (`cert/head/readout_table.py 0hT2rg`, 36 training networks at S42's point): row gain of the bank read: 0 of 104 weights
beyond 3 se, mean t squared 1.28 (no signal); carried b: 24 of 896 beyond 3 (layers 13-14: 10 and 11 of 64), mean t squared
2.10; radial: 9 of 896 (all at layer 14). Line searches on the 16 line-search ranks (`cert/head/solve_rg.py`): carried b,
steps 0.0004 / 0.0012 / 0.0033: -0.092% (se 0.124), +0.75%, +9.85%; radial, 0.025 / 0.075 / 0.2: -0.332% (se 0.269; all layers
+0.21%), +0.35%, +6.75%. Nothing passes.

Final-layer head columns (`otr_m.py` columns 17-19, only at layer 15; column 20 = column 16 on the response rows): certified at
zero weights on rank 3 (forward and gradient run identical to the stored S42 point 1.585407264e-8). Their 36 zero-weight
gradient runs (`a_0hF15g`) and table: `cert/s43/f15.sh`.

The physicist's base module is in the card as `{'module': 'pbase', ...}` (`O/physicist/pbase_frozen_s4a.py`): with every keyword
0 the result is identical; bsk 1 on rank 3: -0.193%.

S43 joint solve (`cert/s43/mix.py`): mixture of arms on the 16 line-search ranks with final predictions kept, base = S42's
arithmetic (`cert/fold/c3q4zd6_optimum.pt` + table law + k31l ages 1-2 + the column at step 2's weights): `bsk` (pbase bsk 1),
`arr` (learned arrays at eta 0.0025 along the `cert/law_ls3` direction), `age3` (k31l cohorts of age 3 at strength 1), `f15` (if
the final-layer columns pass). Then the combined candidate on the 16 ranks for real and ONE judgement on the 50 against S42's
arithmetic (`a_0hS41_u2` on all 50).

Final-layer columns, zero-weight table over 36 training networks at S42's point (`cert/s43/f15.sh`): column 17 (law diagonal at
layer 15) 95 of 128 weights beyond 3 se, mean t squared 30.9, halves 0.99; column 18 (young pairs' plane diagonal at 15) 3,
3.0, 0.63; column 19 (K4L value at 15) 1, 1.65, -0.11; column 20 (K4L value on the response rows, layers 1-14) 21, 4.55, 0.79.
The arm of 17-19 (one step for a first-order -1%) on the 16 line-search ranks: +0.711% (se 0.241, 2/16); the linear model's
best size is a quarter of it, worth -0.09%. Not in S43.

S43 driver `cert/s43/driver.sh` (solve over bsk, arr, age3 -> candidate `cert/s43/s43c.pt` + request -> 16 line-search ranks
`a_0s43c_ls` and the 50 judging ranks `a_0s43c_j` against S42's point `a_0hS41_u2` -> `candidates/s43_mix/`).

Layer-15 dump for the lead's offline fit of the last layer's own correction (`otr_m.py` comment DUMP15; request field
`dump15` = npz path; needs `hf`): H (the head's 126 inputs at layer 15 as gnn_step forms them), Z (the 21 head columns), S_in,
S_out, delta_head, delta_final, host, final, sigma, e2_term, rms15, gain15, ds15. Written, not yet loaded in the workers.

Corrections at 03:13Z (engineer 02:14Z / 03:0xZ): the final-layer columns 17-19 used the PREVIOUS layer's alive mask (other
units); since 03:13Z they use the kept rows of the final layer's row cut, |mu| <= 3.5 sqrt(vv) at layer 15 (`L15_ROWCUT_Z`):
17 regressed over those rows, 18 = my transcription of k31l's plane diagonal (ages 1-2) with the chaos4 means over those rows,
19 = K4L's raw sum with beta over those rows (`project(loc, vv, mu, rows)`; the card's own k4 unchanged). The torch card has no
layer-15 row cut: its pp is real on all rows. The 02:51Z table and the f15 arm (+0.711%) were made with the wrong mask.
New in the same card: batch 6 = the physicist's `O/physicist/b6cols.py` (request `b6`: columns 21-30), readouts o5 / o6 (`rgw`
keys `k4m`, `k3m`: message-based k4 and K21-diagonal k3 before the pair map). All at zero weights: rank 3 unchanged bit for bit
(1.585407264e-8), forward and gradient. One table from 36 gradient runs `a_0z6g` (`cert/head/table6.py`). Version before:
`cert/otr_m_v22_before_kept_rows_b6_o5o6.py`.

S43 JOINT SOLVE JUDGED (10-10 04:05Z; `cert/s43/driver.sh`): arms alone on the 16 line-search ranks against S42's point: bsk
(strength 1) +0.158% (se 0.196, 7/16); learned-array step (eta 0.0025 along `cert/law_ls3`) -0.129% (se 0.046, 11/16); k31l age 3
(strength 1) -0.088% (se 0.201, 13/16). Joint least squares: bsk 0.34, arrays 1.50 x the step (eta 0.0038), age 3 at 0.64; model
-0.339% (leave-one-out -0.216%). The candidate on the 50 judging ranks against S42's arithmetic: **-0.596% (se 0.121, better on
42/50; all layers -0.397%)**; first 25 -0.355% (se 0.183, 19/25); second 25 -0.829% (se 0.146, 23/25). Export:
`candidates/s43_mix/` (candidate.py sha 44b9bab0de19; k31l by cohort age 1.0 / 1.0 / 0.64; bsk 0.34 as a pbase strength).
Column 17 closed (line search + the lead's offline fit: -0.085% held out); layer-15 dump cancelled.

JOINT REFIT (lead 10-10 approval; `cert/refit/jmix.sh A`, `cert/refit/jmix.py`): from S42's point, per step a three-arm mixture
(learned arrays / column 6's weights / batch-6 columns m2 qold eold qb) from 36 exact gradients (step 1: `a_0z6g`; later steps:
fresh batches of the training split), sizes by least squares on the 16 line-search ranks, the new point run for real and accepted
only if better by two standard errors (early stop); then once on the 50, parity predictions on 651 / 378, export to
`candidates/jmA/`.

JOINT REFIT RESTARTED AS jmB (10-10 04:20Z): from S43's point (cert/s43/s43c.pt, bsk 0.34 as the extra pair-map entry, column 6 at S42's
weights; hu cert/refit/s43_hu.json), so that the final point is S44's torch card (jmA had started from S42's point; S43 + jmA would have
taken the learned-array step twice). Four arms per step: learned arrays / column 6 / m2 eold qb (rows 21 28 29) / qold (row 27) alone;
an arm that moves nothing is held at 0 in the solve. Final: once on the 50 with and without qold (lead 03:30Z: qold costs +0.58% of
bill natively), against S42's arithmetic and S43's runs (a_0s43c_j); parity 651 / 378 for both. Certificate: the base runs at S43's
point with the batch-6 hooks on and zero weights reproduce S43's runs bit for bit (ranks 3, 22). FIX (otr_m v24, backup
cert/otr_m_v23_before_b6_layers_1_14.py): qold / eold no longer enter the head at layer 15 (kept rows); batch 6 is layers 1-14 as natively,
f15 alone at 15; four gradient runs made before the fix parked in run_M/parked_z6g_l15qold_1019 and requeued.

jmB STEP 1 ACCEPTED (05:31Z): sizes on the 16 line-search ranks (arms alone at their trial steps): qold -0.621% (se 0.164, 13/16) x1.41;
m2+qb -0.411% (se 0.228, 12/16) x0.98; column 6 -0.060% x0.32; learned arrays +0.354% (S42-point gradients: S43 had taken that direction) x0.21;
model -1.020% (leave-one-network-out -0.796%); REAL RUN of the point against S43's point: -0.997% (se 0.424, 11/16; all layers -0.667%).
jmC (05:38Z) = jmB continued from step 1's point with head row 17 = physicist/t67_U_17.npy (lead 04:15Z) and seven arms (+ column 20, column 17,
kappa_C). kappa_C = dial family kc (otr_m comment kc; backup cert/otr_m_v24_before_kappa_c.py): CERTIFIED against physicist T74 on network 22 at
S42's request, kappa 0.5: 16.205642e-9 both; the gradient mode with the term gives the same MSE (1.620564192e-8) and a kc gradient.

jmC STEP 2 ACCEPTED (06:45Z): base = step-1 point + column 17 (physicist's weights): column 17 alone -0.327% on the 16 (se 0.130, 11/16).
Arms (live: arrays, m2/eold/qb, qold, column 20, kappa_C; column 6 and column 17 at noise level, not run): kappa_C -0.353% alone (11/16) -> kc
0.303; arrays -0.151% -> 2.11 x trial; column 20 -0.281% -> 1.03; m2/eold/qb +0.020% -> 0.34; qold +0.407% -> -0.20. Model -0.944% (LOO -0.597%).
REAL: -0.989% (se 0.232, 12/16; all layers -0.956%). Running total on the 16 against S43: 1.4128e-8 / 1.4460e-8 = -2.30% (sizes fitted on the 16).
LEAK found and fixed 06:50Z: step 2's gradient batch held judging ranks 277 and 480; from step 3 batches exclude the 16 and the 50 (jmix9.sh);
primary final judgement on the 48 clean judging ranks (cert/runM_val48_clean.txt).

jmC STEP 3 NOT ACCEPTED (07:24Z; -0.529% on the 16, se 0.345: below two standard errors) -> step 2's point stands (cert/refit/jmC_p2.pt + _hu.json).
**jmC JUDGED ONCE (07:48Z), 48 judging ranks never in a gradient batch (cert/runM_val48_clean.txt), torch forward:**
  WITHOUT qold (row 27 zeroed): -2.068% against S42's arithmetic (se 0.254, 44/48; all layers -1.42%); 50 ranks -2.006% (se 0.252, 45/50);
  first 25 -1.50%; against S43's runs -1.416% (se 0.223, 40/48).
  WITH qold: -2.001% (se 0.318, 39/48); qold's own part +0.068% (se 0.172, 24/48): not carried (+0.58% bill natively).
Hand-over candidates/jmC/: candidate.py sha 68b386ac6dc7; physics_dials.json (kappa_C 0.697 at layers 1-14; k31l 1/1/0.64);
head_weights_31x128_without_qold.json sha 4e5b245d8c7d (rows 6 17 20 21 28 29, norms 0.39 0.72 0.55 0.20 0.02 0.30); parity a_0mCparq_r0651/0378.

jmD (07:54-09:06Z, cert/refit/jmix10.sh): continuation from S44's torch point (jmC without qold), sized and accepted on a FRESH 16
(cert/refit/sz16_ranks.txt), kappa_C split 1-9 / 10-14, qold out (its gradient stayed strong: 89 of 128 beyond 2.5 se). Step 1 ACCEPTED: -0.659% on
the fresh 16 (se 0.195, 13/16; arrays 2.14 x trial, kappa_C 0.62 at 1-9 / 0.92 at 10-14, column 6 0.43, column 20 0.21); step 2 not (-0.337%, se 0.193).
JUDGED ONCE on the 48 clean: -0.455% against S44's torch runs (se 0.131, 35/48), -2.513% against S42's arithmetic (se 0.260, 44/48).
Hand-over candidates/jmD/ (candidate.py sha 176a46113595; head sha 0ce065ebea93; parity a_0mDpar_r0651/0378). export_dials.py now carries a
per-layer kappa_C (backup cert/export_dials_before_kc_per_layer.py).
jmE (09:10Z, cert/refit/jmix11.sh): banded head weights (otr_m HSPLIT, rows 31 / 32 = columns 6 / 20 at layers 10-14; backup
cert/otr_m_v25_before_hsplit.py), from jmD's point, second fresh sizing 16 (cert/refit/sz16b_ranks.txt). Certificate: banded start = jmD's point
within 2e-5 relative final MSE on 453 / 268.

jmE CLOSED (09:27Z): banded column weights carry no signal (rows 31 / 32 = columns 6 / 20 at layers 10-14: 0 of 128 beyond 2.5 se, mean t^2 0.70 / 0.50
over 31 networks); the step on arrays + kappa_C early / late gave -0.313% on the fresh SZ16b (se 0.218): not accepted; jmD's point stands.
jmF (09:28Z, cert/refit/jmix12.sh, JM_REQ {"hbands": 1}): band biases of the head's first layer (otr_m HBANDS rows 33 / 34 / 35 = constant 1 at layers
1-9 / 10-14 / 15; backup cert/otr_m_v26_before_hbands.py); zero weights bit-identical to jmD (294, 153). Zero-weight gradients over 32 networks (batch 6):
rows 33 / 34 / 35: 50 / 46 / 59 of 128 beyond 2.5 se, mean t^2 5.35 / 6.05 / 7.75, split-half cosine +0.57 / +0.74 / +0.88. The same batch shows an
unusually strong common array gradient (42% of elements beyond 2.5 se; batches 3-5: 2-4%); checked: same network = same gradient under both
configurations (419: 1e-4; 490: 1e-3), the direction agrees with batches 4-5 at the same point (cosine 0.66-0.90): a batch effect, not a bug.

jmF JUDGED (10:29Z): step 1 (band biases + arrays + kappa_C 0.47 / 0.67) on the 48 clean: +0.203% against jmD (se 0.203, 21/48); -0.253% against S44.
Its acceptance (-0.939% on sz16c, se 0.324) was on the set that fitted the sizes: biased. jmD remains the S45 refit increment.
jmG (10:31Z, cert/refit/jmix14.sh): from jmD's point with batch 7 (otr_m B7 rows 36-41 = fold kmu kg qk e34 s1 over all units; backup
cert/otr_m_v27_before_b7.py; certificate: zero weights bit-identical on 294 / 153); sizes on sz16d, ACCEPTANCE on the independent sz16e at one se;
final judgement with and without fold (row 36).

jmG steps (independent check on sz16e, 1-se bar; sizing on sz16d): step 1 ACCEPTED -0.408% (se 0.389, 9/16; sizing set read -1.151%): fold 2.79,
m2/eold/qb 0.87, kmu+kg+qk 0.18, e34+s1 -0.21, arrays 0.21, column 6 0.27, kappa_C 0.73 / 0.82; step 2 ACCEPTED -0.352% (se 0.127, 11/16; arrays
0.60, small kappa_C; sizing set -0.034%); step 3 ACCEPTED -0.312% (se 0.134, 12/16; arrays 2.15). b7 zero-weight table at jmD's point (30 networks,
beyond 2.5 se / mean t^2 / split-half): fold 90 / 12.5 / +0.98; kmu 22 / 3.5 / +0.84; kg 23 / 3.5 / +0.87; qk 24 / 3.6 / +0.85; e34 18 / 2.5 / +0.15;
s1 25 / 3.7 / +0.09.

jmG step 4 not accepted (+0.016% on sz16e). **jmG JUDGED (48 clean)**: with fold -0.610% against jmD (se 0.205, 34/48), -1.062% against S44;
without fold -0.254% / -0.707%; fold's part -0.358% (se 0.196) for +1.34% of bill natively: out. With ALL batch-7 rows zeroed: -0.236% against jmD
(se 0.103), **-0.690% against S44 (se 0.157, 35/48)**; the cheap five rows' part -0.018% (se 0.020): out. S45 REFIT CANDIDATE = jmG without batch 7
(same native parts as S44): candidates/jmG/candidate.py sha a210c114372c, physics_dials.json (kappa_C 0.701 / 0.885), head_weights_31x128_without_b7.json
sha 94f2342d28d2; parity a_0mGparn_r0651 / 0378.

ZERO-DIAL SCREEN at jmG's point (13:30Z; 36 runs a_0cvg; cert/cv/dial_table_0cvg.out): low-rank covariance corrections (otr_m cvdd ... cvmm,
backup cert/otr_m_v28_before_cv.py) at noise: CLOSED. Old dials off: X1 13/14 layers beyond 2.5 se, mean t^2 17.1; bscale 6.3; c0 4.6; k31 4.3.
jmH (cert/refit/jmix15.sh; dial arms = uniform shifts; sizing sz16f, independent check sz16g): step 1 ACCEPTED -0.843% on sz16g (se 0.326, 13/16):
X1 +0.094 (alone -0.30%, se 0.08, 14/16), k31 +0.16 / c0 -0.07, kappa_C 0.50 / 0.77; step 2 ACCEPTED -0.464% on sz16g (se 0.203, 11/16; arrays
1.74, column 6 0.77, X1 +0.4 x trial, k31 / c0 again opposed).
PRIMARY JUDGE CHANGED (lead 13:47Z): the 32 untouched generated networks (physicist t79, seeds 7100060-7100091); jmG without b7 there: -0.39%
against S44 (se 0.20, 21/32), against -0.69% on the 48 clean (selection on the 48).

jmH step 3 not accepted; **jmH JUDGED on the 48 clean (secondary)**: -1.072% against jmG without b7 (se 0.222, 35/48), -1.754% against S44 (se 0.222,
43/48), -3.79% against S42. Dials: X1 0.557 -> 0.708, c0 0.534 -> 0.517, k31 -0.212 -> +0.107, bscale 1.038 -> 1.036, kappa_C 0.49 / 0.64.
candidates/jmH/ (candidate.py sha 4eadd228e719, head_weights_31x128.json sha 72fdc0da057c); parity a_0mHpar_r0651 / 0378. Primary judge (32 untouched
seeds, physicist t79) requested.

**jmH ON THE 32 UNTOUCHED NETWORKS (physicist t79, 15:10Z): -1.33% against S44's torch point (se 0.31, 25/32), -0.94% against jmG (se 0.35, 23/32).
FIRED AS S45 SAFE #335213 (lead 15:11Z; native sha b36b7154, ten fresh -1.48% against native S44, bill +0.000%).** Dial re-screen at jmH's point:
no global family off (largest mean t^2 3.4); late-band sums only (X1 -3.7, iso -3.4, bscale +3.0, k4la1 +3.1, s_carry -2.7).

jmI (15:24-17:07Z; late-band dial arms + arrays + head + kappa_C; sizing sz16h, check sz16i): step 1 -0.610% on sz16i (se 0.140, 15/16), step 2
-0.356% (se 0.171, 11/16). On the 48 clean: full jmI -0.821% against jmH (se 0.116, 40/48); NATIVELY FREE FORM jmIf (X1 / bscale / s_carry / iso
back to jmH's): -0.501% (se 0.132, 32/48; -2.25% against S44); jmIv (+ iso +0.030 at layers 10-14, needs per-layer _GN_KNOBS): -0.738% (se 0.139,
38/48). Uncapped first-order trial shifts of flat dials blew up in step 2 (+83% / +195% alone): dial trial shifts now capped at 0.05
(JM_DIAL_CAP). Hand-over candidates/jmIf/ (candidate.py sha b62b4e42b4e1, head sha b74ad6e505bf). Worker field 'sched' (old-tier join ranks)
certified against physicist T80 (534, both schedules, 12 digits). jmJ: refit on the 384-384-384 base from jmIf (sizing sz16b, check sz16).

jmJ (18:00-19:20Z, refit on the 384-384-384 old-tier base from jmIf): fixed-parts effect of 384 at jmIf on sz16 -1.729% (se 0.371, 15/16);
step 1 NOT accepted (+0.228% on sz16, se 0.148; LOO +0.31%): no co-adaptation gain; the 384 base nets about zero against +1.68% bill.

10-10 evening: POOL OUTAGE 19:30-21:06Z (supervisor start cap 30 per index; first replacement lacked TRAINER_PL=1 TRAINER_CARD=M: 173 requests failed,
parked in run_M/errors_noPL_2054, requeued). STALENESS SCREEN at jmIv (36 runs a_0frzg, unfreeze; cert/cv/frozen_table_0frzg.out): September-frozen
centers / scales / rms / gain / first at noise; feature statistics normal (otr_m FEATS); the learned EDGE e.* shows strong zero-step gradients (rho_scale
mean t^2 27, bias 10.7, output 8.3, left 7.1, edge 6.4). jmL (JM_ARR_SPLIT=1, edge arm separate): edge arm alone +0.447% at its trial step; NO STEP
(+0.052% on sz16d). jmK stopped half-way (parked run_M/parked_mK1_for_jmL_2200). Lane converged at jmIv = S46 (fired by the lead).

## HANDOVER NOTE FROM THE CLAUDE TRAINER (2026-10-10 23:45Z; the lane passed to Astra at 23:26Z, see TAKEOVER_ASTRA.md)
Request fields added today (worker.py / otr_engine.py / otr_m.py; each off by default, each certified bit-identical when off):
  'sched' (old-tier join ranks L7-L10-L13, default 384-320-384; certified vs physicist T80 on 534); 'unfreeze' (September-frozen p.centers /
  scales / rms / gain / first get gradients and may differ); 'featstats' (head-feature mean / std / clip share per layer); 'lossw' (all-layer
  training loss; the line was closed 09-24); 'cv' (low-rank covariance corrections computed at 0; closed); 'b7' (physicist's batch-7 columns,
  rows 36-41; closed); 'hbands' (band biases rows 33-35; closed); 'hsplit' (banded column weights rows 31-32; closed); dial family 'kc' (kappa_C).
cert/refit/jmix.py env: JM_SIZING (sizing ranks file), JM_KC_SPLIT, JM_DIALS ('fam' or 'fam:lo:hi'), JM_DIAL_CAP (0.05), JM_ARR_SPLIT (head vs edge
  arrays). Drivers: jmix15.sh is the last one I used (env JM_REQ, JM_PREV, JM_CHECK, JM_G1, JM_DIALS; args tag params hu k1 k2 first-batch extra-ncar
  sizing-file). Pool: supervisor needs TRAINER_PL=1 TRAINER_CARD=M; 30 starts per worker index.
Converged at jmIv = S46 on every screened knob; the open lever named by the outgoing lead is the cut-simulated refit (engineer a22_cutsim).

## Nd (10-11 03:50-06:35Z, lead task Nd; trainer back from the Astra interval)
Nc's 96-gradient arm set (12 arms, cert/refit/jmNc_d1) + a 13th arm dds_15_15 (layer-15 correction size, lead 04:15Z) evaluated on 7100152-167
(run_Nd, Nc's exact cut-aware worker config, two guarded workers) and re-sized by ridge on 32 networks 7100136-167 (cert/Nd/solve_ridge32_v3.py):
arm set chosen by LOO before any check = without kappa (-0.462% at ridge 0.2; all 13 -0.423%); model -0.669%. Group LOO alone / full-without:
arrays -0.24 / -0.30, head -0.27 / -0.21, kappa +0.003 / -0.46, dials -0.06 / -0.38, layer-15 scale +0.001 / -0.43 (%). Frozen point
cert/refit/jmNd_p1.pt sha 4f066ac5b358 (+ hu 6a2d7817ce0e), record cert/Nd/FROZEN_POINT_v1.json. **CHECK on fresh 7100232-263 (A30):
-0.482% against S46 (se 0.205, better on 19/32; all layers -0.215%): ACCEPTED** (cert/Nd/CHECK_VERDICT_v1.json). Hand-over candidates/jmNd/
(candidate.py sha b0d9b738fed5 from cert/refit/jmNd_p1_exportable.pt = iso as one value; real per-layer iso via the .pt; head sha 03794864b808);
parity run_Nd/results/a_0mNdP_{cut,plain}_r0651/0378.json. Certificates: Astra's point tool rebuilds jmNc_p1 byte-identically; my solver reproduces
Nc's 16-network solve exactly; the S46 base on 152-167 equals Nc's check runs bit for bit.

## Ne (10-11 06:36-10:05Z; option (a): sizing on 7100232-263)
96 fresh public gradients at Nd (outside Nc's 96 and the 16 + 50; cert/Ne/grad96_ranks.txt); t-table cert/Ne/band_table_v1.out (head arrays t^2 4.3-4.6,
k31 / iso / X1 11-14 / ds 7-10 / s_carry 1-6 t^2 9-11, e.cov_factor 18). 11 arms (cert/Ne/arms_v1; tool jmix_generated_v5.py, env arms_env_v2.sh) x 32 on
7100232-263. Solve (cert/Ne/solve_ridge32_ne.py): every LOO >= 0 (best +0.017% at the largest ridge); FROZEN point = Nd within rounding
(cert/Ne/FROZEN_POINT_v1.json, jmNe_p1 sha 6d92b0b087ce). Arms alone at trial: h17 +2.46%, ecov +0.62%, hb3 +0.54% ... best X1 11-14 -0.04%.
The public-rank gradients at Nd do not transfer to generated networks. A36 check held (lead's call). Workers stopped (run_Ne/STOP).

## Nf (10-11 10:05-11:00Z): gradients from GENERATED networks (A36 7100264-327, 16M truths; otr_generated_v2 / worker_generated_v5 / jmix_generated_v6)
Generated vs public gradient at Nd (cosine | t^2 public / generated): population signal (same direction, both strong) in c0, k31, iso 11-14, h17, eold,
edge scalars; public-specific: head arrays (generated split halves negative), s_carry 1-6, ds 7-10, X1 11-14. Ne's population arms had overshot
(first-order trial steps sized on public slopes). Nf: 7 arms with capped trials x 32 on 7100232-263: every LOO >= 0; frozen point = Nd
(cert/Nf/FROZEN_POINT_v1.json); arms alone within +-0.01% except eold+qb +0.14%. REFIT ENGINE CONVERGED AT Nd on generated data; check not run.

## Ng (10-11 11:05-11:35Z): new per-unit INPUTS of the learned edge, zero-weight gate on 64 generated networks (lead's research bet 1) - FAIL, closed
otr_m.py comment EDGE INPUTS (request field 'edgein', worker_generated_v6.py, record 'eu_grads'): 10 columns at layers 8-14, each with 32 zero weights
into the edge's left / right pre-activations (the card's own translation of _message_edge_source with + context['zl'] / ['zr']). Identity at zero
weights: final / all-layer MSE and all 84 gradient entries bit-identical to run_Nf on 7100264 / 265 (cert/Ng/identity_check.py). Table on A36
7100264-327 at Nd (cert/Ng/edge_table.py, _v1.out), shared over 8-14: W'd/s 2.42 (halves cos +0.41), law diagonal 2.40 (+0.29), qold 2.08 (+0.51),
eold 1.06, K4L 0.82, K21 row mean / col mean / row rms 0.61 / 1.18 / 0.68 (strong in half A, sign flips in half B), rho row mean / rms 0.67 / 0.80;
existing edge inputs at the same point: features median 1.01, messages median 0.84. Rule (mean t^2 > 4, halves cos > 0.5): none pass. run_Ng/STOP.
