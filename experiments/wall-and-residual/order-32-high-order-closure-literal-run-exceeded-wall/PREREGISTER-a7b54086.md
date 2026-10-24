# Full V4 Q24 wall-safe closure successor

Status: **target-free additive source construction; no target, score, dataset,
sampling, fit, submission, graph mutation, or scientific promotion is
authorized by this file**.

## Question and immutable parent

Can the frozen full V4 analytic closure retain its complete CP3, H3, CP4,
H4+alpha, L4, and `L3^2/2` science while reducing the slow Gaussian pair
kernel enough for robust Phase-2 board wall time?

The sole source parent is
`../fullbudget_k21_k4_closure_v4/estimator.py`, 68,698 bytes, SHA-256
`3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c`.
Its literal local meter passed at `2,179,742,954,469` FLOPs, 35.341105 seconds
wall, 34.358 seconds of backend work in the common meter, 0.392632 seconds
residual, and 4,618,567,680 bytes descendant RSS.  Submission 328199 then
showed two completed e-8-class rows but failed operationally on board wall
time.  The CP4-off A1 contrast subsequently passed its local meter yet made
backend time slower, about 35.072 seconds, so fast CP4 GEMM deletion is not a
credible board-wall remedy.

This arm is the already target-free-designed `D_Q24_Q32` contrast in
`../fullbudget_k21_k4_closure_accuracy_frontier_v1/PREREGISTER.md`.  No result
from a target or board row selected Q24.

## Single scientific change

Q24 means the inclusive Hermite/Mehler sum `q=0..24` (25 terms), replacing
V4's inclusive `q=0..32` (33 terms).  Exactly these payloads shrink:

- inverse factorials become `1/q!` for `q=0..24`;
- each of the seven derivative-order gather rows becomes `d..d+24`;
- the persistent rho-power arena has 25 planes and its metered cumulative
  product constructs `rho^0..rho^24`; and
- every pair contraction uses that 25-entry term axis.

The Q38 coefficient generator stays at `q=0..38`, so derivative features up
to order 30 required by Q24 remain available without changing their formulas.
All 17 derivative channels remain present in their frozen order: baseline,
four L3, five L4, and seven `L3^2/2`.  Their weights, six sequential
three-channel accumulation groups, dtypes, operation association, CP3, H3,
CP4, H4+alpha, PSD repair, W16 specialization, admission guard, and every
other closure state are unchanged.

## Resource projection

Pinned FlopScope 0.12 whole-source identity meters at widths 4, 5, 6, 8, 12,
and 16 give the exact parent-minus-child polynomial
`46,304*n^2 + 155,904*n + 16`, with identical operation-call counts at every
width.  It projects as follows at production width 1024 and depth 16:

| source | FLOPs |
|---|---:|
| frozen V4 literal parent | 2,179,742,954,469 |
| projected whole-source removal | 48,712,908,816 |
| projected Q24 child | 2,131,030,045,653 |
| headroom below `2^41` | 67,993,209,899 |

Thus the pre-meter projected bill is 97.765% of the V4 bill and 96.908% of
the Phase-2 budget, safely below `2^41`.  The older frozen frontier recorded only
the dominant pair-einsum delta (`48,082,370,560`) against its earlier
execution spelling; the literal current-source meter supersedes both
projections.

The certified V4 diagnostic measured 26.004235 seconds of 33.148912 backend
seconds in einsum and another 3.241066 seconds in the 32-power cumulative
products.  The Q24 term-axis ratio is `25/33`; the cumulative-product work
ratio is `23/31`.  Even without credit for smaller gathers and temporaries,
the expected local backend is about 26--28 seconds.  Scaling by the observed
board/local backend ratio gives a provisional board predict range around
95--104 seconds, versus 119 seconds for V4.  This is a preregistered
engineering estimate, not a wall claim; only an identical-driver literal
meter and a separately authorized board attempt can establish it.

## Target-free qualification gates

1. A mechanical builder must bind the exact V4 bytes and emit one additive
   estimator.  A normalized whole-module AST comparison must prove that the
   only executable differences are the four Q-order payloads above; truthful
   names and docstrings may also change.
2. An independently spelled Q24 reference made by slicing V4's Q32 term axis
   must be bitwise equal on six fixed depth-16 synthetic fixtures: the four
   canonical V4 fixtures plus the A1 alternate identity-plus-QR and
   `0.88 I` contractive fixtures.  All
   outputs must remain finite and every production/off-shape guard must be
   unchanged.
3. Record, but do not tune against, Q24-versus-Q32 layerwise prediction
   disagreement.  The frozen frontier's causal gate is final prediction
   disagreement MSE above `0.01 * min(selected_final_raw_mse, 1e-8)` or a
   changed moment-feasibility verdict.  Synthetic disagreement is supporting
   evidence only and cannot substitute for the fixed-roster gate.
4. Pinned client/server API smoke must exercise the 25-plane cumprod `out`
   slice and the explicit all-17 batched einsum.  No NumPy, compiled kernel,
   worker, computation in flight, or residual arithmetic may enter deployed
   source.
5. One coordinated literal meter must return finite float32 `[16,1024]`, at
   most `2^41` FLOPs, wall below 120 seconds, residual at most 0.4 seconds,
   and descendant RSS below 8 GiB.  Robust wall qualification additionally
   asks for local backend at most 30 seconds and residual at most 0.30 seconds.
6. Freeze, independent audit, packaging, accuracy access, and any board action
   are separate authorities.  No same-archive retry or target adaptation is
   permitted by this design.

## Outcome law

- If source/reference/resource gates pass, freeze Q24 as one target-free
  wall-safe candidate for the separately preregistered fixed-roster accuracy
  comparison against full V4.
- If Q24 changes synthetic/reference science beyond the exact truncation or
  fails resources, reject these bytes; do not tune the order.
- If its fixed-roster prediction disagreement is negligible and resource
  gates pass, Q24 is the preferred operational spelling of the same closure
  at measured resolution.
- If disagreement or accuracy is materially adverse, retain full V4 as the
  accuracy anchor and activate the already preregistered A2 third-only wall
  arm; do not search intermediate Q orders.

This experiment does not establish raw MSE, generalization, moment
feasibility, ranking, or board wall time before their explicit gates run.
