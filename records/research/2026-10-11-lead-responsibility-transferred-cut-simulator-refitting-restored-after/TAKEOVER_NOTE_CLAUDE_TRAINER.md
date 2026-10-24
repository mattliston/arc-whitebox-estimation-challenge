# Trainer lane, handover note from the outgoing Claude trainer (2026-10-10 23:50Z)

TAKEOVER_ASTRA.md, written by Astra, holds the authoritative state. This note only adds the exact commands from the outgoing trainer.

RUNNING: nothing of mine. jmL is TERMINAL (23:29:58Z): no step; the edge arrays sit on sharp curvature. The pool (supervisor + 2 workers) is idle and inherited by Astra.

POOL. Start it from O/trainer:
  bash ../gate.sh 14 trainer-pool && TRAINER_PL=1 TRAINER_CARD=M nohup /Users/matt/anaconda3/bin/python -B supervisor.py run_M 2 4 >> run_M/workers/supervisor.log 2>&1 &
- Without TRAINER_PL=1 TRAINER_CARD=M every request fails at once. Check the start line in run_M/workers/w0.log: "card": "M", "per_layer": true.
- The supervisor starts each worker index at most 30 times. Reload after code changes: echo 0 > run_M/WORKER_COUNT, wait until no worker is left, then echo 2.
- Watch run_M/workers/supervisor.log after every reload.

REQUEST FIELDS (worker.py). S46's own request: f4c 1, ss 0, hf 2, b6 1, ncar [n6y dfit 2, k31ly ages 1/2/3 with leaves nkl1-3, pbase bsk 0.3402257055212658], hu = the 31-42 x 128 head table.
Optional fields, all off by default and bit-identical when off:
- sched: old-tier join ranks, e.g. '384-384-384'.
- unfreeze: the September-frozen p.* arrays get gradients.
- featstats.
- lossw: closed.
- cv: closed.
- b7: rows 36-41.
- hbands: rows 33-35.
- hsplit: rows 31-32.
Dial shifts travel in the params file's pl (otr_pl families + kc = kappa_C).

REFIT ROUND (from O/trainer; driver cert/refit/jmix15.sh, step tool cert/refit/jmix.py):
  JM_ARR_SPLIT=1 JM_REQ='{}' JM_PREV=<start point's label on the 50, e.g. 0mIJv> JM_CHECK=<independent check ranks file> JM_G1=<existing gradient label at the start point, or leave unset> \
  JM_DIALS=X1,bscale,c0,k31 nohup bash cert/refit/jmix15.sh <TAG> <params.pt> <hu.json> 1 <max steps> <first batch index> '[{"module": "pbase", "bsk": 0.3402257055212658}]' <sizing ranks file> >> cert/refit/jmix_<TAG>.log 2>&1 &
- A dial entry may be fam:lo:hi to shift one band only. Trial shifts are capped by JM_DIAL_CAP (0.05).
- PROTOCOL:
  - Size the step on one fresh 16.
  - Accept on a DIFFERENT fresh 16 at one standard error. A check on the sizing set is biased: jmF passed there and failed held out.
  - Gradient batches must exclude cert/ls16_ranks.txt, cert/runM_val50.txt and all sizing / check files (cert/refit/sz16*_ranks.txt).
  - Training-rank sets are nearly used up; Astra's generated-seed support (worker_generated_v3.py) is the way on.
- The driver judges the final point once on the 50 (labels a_0m<TAG>J). Primary number: python -B cert/fold/pairs.py cert/runM_val48_clean.txt <label>:<ref>. Then the physicist's t79 on untouched seeds decides.

EXPORT: python -B export_dials.py <params.pt> candidates/<name> "<note>", then copy the hu json into candidates/<name>/head_weights_31x128.json (rows 31+ must be zero).
- export_dials refuses per-layer shifts of global dials and knobs (X1, bscale, s_carry, iso).
- For those, export from a copy with those shifts set back (as cert/refit/jmI_p2_free.pt). The engineer reads per-layer values straight from the real .pt (O/engineer/a52_pld/dly_from_pt.py); S46 was built that way from cert/refit/jmI_p2_noAS.pt.

OPEN LEVERS, in the outgoing lead's order:
1. Refit on the cut-simulated card (engineer a22_cutsim). Certify it against native predictions on 651 / 378 first; Astra's v7 failed fidelity at 23:37Z.
2. The b-fix gamma (0.1 -> 0.15-0.2) as a dial arm, about -0.2%.
3. Re-screen every dial family by zero-step gradients after each array move (cert/cv/dial_table.py <grad label>). X1's drift gave S45.

CLOSED today (measured): qold, fold, the batch-7 columns, banded column weights, band biases, low-rank covariance corrections, the 384 old-tier base, the frozen p.centers / scales / rms / gain / first, the learned edge arrays at a finite step, layer-15 refits, the multi-layer loss.
Record: RESULT.md, with its handover section at the end. Memory: p2-1010-joint-refit-jmc.md.
