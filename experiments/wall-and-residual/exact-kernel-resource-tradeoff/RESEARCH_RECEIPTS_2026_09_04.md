# Fresh Phase 2 research receipts — actual date 2026-09-04

All development raw MSEs below are width1024/depth16 ranks35 and100,
respectively, against stored 1e9 post means. These are development screens,
not the independent14-net promotion gate. No frozen historical work was
edited, no promotion or commit performed. Dates use the clock and mtimes.

## New independent card

The organizer SIMPLE K3 map was independently implemented in
`k3_radial_numpy.py`, then mechanically ported to FlopScope. Float64 moment
arithmetic with float32 carrier products retains accuracy; the historical
all-float32 failure was cancellation in the moment arithmetic. Full-covariance
CP births are an exact gauge change relative to off-diagonal births when all
repeated entries are restored; this makes one-star truncation much better.

Unlearned identity-radial host: 4.0824e-8 / 4.3204e-8.
Covariance-metric fourth cumulant: 3.2346e-8 / 3.2068e-8.
Full covariance host learned on 200 honest rows, fixed160/40 net split,
40epochs, C64, mean head only: 2.7374e-8 / 2.6161e-8 in NumPy.
Held-out final one-step MSE 7.554e-9 versus host9.229e-9 (18.15% removal).
Feature arrays are captured before current-layer mean correction; no v-head.

`candidate_radial_gnn_splitread_v1.py` uses only FlopScope numerical ops.
The final oversized CP read is split algebraically into D4-compatible blocks.
Rank35 raw2.73668e-8, local utilization0.340816 (0.352039 without splitting).
Official-version local 35/100/35 gate passed. NumPy output maximum deviation
2.19e-6, learned correction maximum deviation1.35e-8; repeated rank35 output
byte-identical. PeakRSS7.22GB. Residual time0.394s cold with diagnostic copying,
0.322s second call with copying, 0.330s hot clean call; the cold margin is small.

Full independent4M, noise-corrected panel:

| Panel | Raw | Utilization | Adjusted | Versus recovered329790 card |
| --- | ---: | ---: | ---: | ---: |
| Public8 | 3.38836e-8 | .3408 | 1.15481e-8 | -28.04% |
| Virgin6 | 3.01877e-8 | .3408 | 1.02885e-8 | -20.95% |

## What the next state needs

Frozen honest teacher-forced rows plus two independent32M cumulant label
streams show that supplying true pre variances removes60–63% of late-layer
one-step error on both development nets. Supplying true k3 alone removes
~13–15%; true k4 alone~17%; all three reduces late one-step MSE to~1.4e-10.
This is an oracle diagnostic, not a usable estimator or training feature.
Receipt: `radial_state_attribution_v1.json`.

## Bounded screens

- Keep512 newborn CP terms, preserve dropped repeated entries in star:
  raw~9.9e-8 to1.36e-7, versus3.2e-8 full. Closed this pruning variant.
- Gaussian Mehler order6:3.202e-8 /3.201e-8, effectively wash.
- Full first-order Edgeworth pair corrections through correlation order2:
  3.086e-8 /3.659e-8 with one-star host. No consistent gain.
- Fourth metric C+beta*m*m^T, beta=.25 or1:1.09e-7 to2.34e-7. Rejected.
- Sym(C,D) fourth closure matching every partial trace: carry1 gives
  3.109e-8 /3.016e-8; carry0 gives3.189e-8 /2.934e-8. Small gains only.
- One cubic factor per birth became unstable. Rejected.
- Two signed cubes per birth cancel the spurious pure-covariance cube
  algebraically and retain the leading coordinate-star component:
  analytic beta gives2.797e-8 /2.945e-8. Fitted beta worse3.137e-8 /3.263e-8.
  Current repeated entries remain exact; discriminant clipping leaves a
  residual star. Tiny algebra certificate passed. No learned head yet.
- Compress only the subtractive cube component into top32/top64 covariance
  modes, restoring current repeated entries:1e-6 to1e-5 raw. Rejected.
- Keep all coordinate stars in the covariance-metric host (accuracy-only,
  roughly double carrier):2.585e-8 /2.601e-8.
- All-star plus full Edgeworth pair map:2.280e-8 /2.861e-8.
  Two-cube plus full pair map:2.419e-8 /3.029e-8. No consistent late gain.
- Natural K4-to-K3 quadratic-vertex source, represented by an extra
  covariance-vector cubic plus corrected CP birth coefficient:
  3.146e-8 /2.890e-8. Modest gain, not coadapted or metered yet.

There is no claimed universal impossibility. These receipts close the tested
implementations, not all representations. The ~4e-9 goal is still unmet.

## Later September 4 PDT: two-cube coadaptation and boundary audit

The two-cube analytic-birth host was coadapted on 200 honest rows using the
fixed 160/40 cross-net split, C64, 40 epochs, mean head only. Held-out final
one-step MSE is 6.01057e-9 versus host 7.685e-9 (21.79% removal). Closed-loop
development raw is 2.21231e-8 / 2.11385e-8, against the stored 1e9 means.
Payload: `payload_binary_radial_v1.npz`; rows: `rows_binary_radial_v1`.

The full independent4M noise-corrected panel for
`candidate_binary_learned_v1.py` gives public8 raw2.503e-8, utilization.3407,
adjusted8.527e-9; virgin6 raw2.231e-8, utilization.3407, adjusted7.601e-9.
This is a local competitive result, not an official graded score.

Important qualification of the repeated-entry claim above: the analytic
birth uses a clipped beta/lambda coordinate shift but its off-diagonal plane
formula still uses the unclipped beta. Exact equality at clipped units has
not been certified. This frozen version is not silently changed; the stated
tiny algebra certificate applies to the unclipped representation.

The diagonal-only covariance feedback variant was regenerated and retrained
honestly, not tested with an unmatched head. Held-out one-step final MSE is
4.79260e-9 versus host5.774e-9, but closed-loop development raw is
2.27396e-8 / 2.04621e-8: mixed versus the full-feedback model. Not promoted.

Two graded submissions have different implications:

- 329790, recovered historical CP3-8192 payload: adjusted1.4369934e-8,
  raw3.1560041e-8, 10.84% better than official329754. Current official best.
- 329798, initial new radial learned card: adjusted1.7476524e-8,
  raw3.1709440e-8. Its official utilization was .55115, not local .3408.
  Root cause was our native-only dtype gate: remote `.dtype` is a string,
  so comparing it directly with `fnp.float32` disabled every fast product.
  All newer learned binary candidates compare `str(dtype)` instead. The
  fixed radial candidate passed the actual numpy-free client/server gate
  at utilization.340831, residual.16444s. See the separate dtype-bug receipt.

## Exact-kernel resource tradeoff

Depth-six rectangular products with 8192-wide tiles preserve the full panel:
public8 adjusted7.798e-9, virgin6 adjusted6.952e-9, utilization.3115. Actual
client/server utilization.311646 and residual.22165s passed, but native
35/100/35 peak RSS reached9.433GB, above8GiB. Do not submit this version.

Halving tiles to4096 costs only utilization.312745 at the client boundary.
Native head parity and byte-identical repeated output pass, peak RSS7.373GB.
Under concurrent load its actual client residual was.42258s, above.4s;
do not declare a complete resource pass from native results alone.

Depth-five, 8192-wide tiles are the current resource-safe alternative.
Actual client/server utilization.322554, wall49.14s, residual.33626s under
concurrent load. Native first-call peak RSS7.229GB, learned-delta maximum
deviation6.64e-8 versus NumPy. Final repeated-call and 14-net checks ongoing
at the time of this entry. The older D4 full-bank candidate itself reached
8.914GB peak native RSS, so it is not an unconditional resource-safe fallback.

## Current falsifiable cost round

Signed pairs must be selected together. Their self energy is
`a^2 * (||p||^6 + ||q||^6 - 2*(p.q)^3)`, and only dropped pairs are read to
restore the current repeated planes. Unlearned development raw:

| Pair cap | Rank35 | Rank100 |
| --- | ---: | ---: |
| Full | 2.79718e-8 | 2.94529e-8 |
| 6144 | 3.56217e-8 | 4.08371e-8 |
| 4096 | 6.25638e-8 | 6.98717e-8 |

These unadapted host results do not judge cap-specific learned models.
One bounded cap4096 retraining round is now running; 200 rows completed,
honest one-step profile L10=1.13042e-8 to L15=1.79273e-8. The hypothesis is
that the additional truncation error is sufficiently predictable for the
matched closure to recover near-full accuracy at much lower utilization.
Judge on closed-loop development first, then both independent panels.
