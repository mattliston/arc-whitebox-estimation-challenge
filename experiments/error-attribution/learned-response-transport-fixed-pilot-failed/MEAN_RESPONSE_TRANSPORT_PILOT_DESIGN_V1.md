# Plane-preserving response transport, fixed matched pilot

September6,2026, Phase2. Prerequisite gate-information verdict passed on
both rank35/100: learned/Gaussian cross-MSE .019153/.017730, learned/host
.872946/.858712. The paired .90-host gate margin is only .68/.89
jackknife standard errors, so this is an exploratory information pass,
not a high-confidence >10% advantage over host and not a final score.
No more gate streams or seed/layer choices are allowed.

Change inherited CP legs ONLY at predeclared layers8..14. Let q be the
clipped location derivative of host+frozen bank-P0 correction. Replace
g_i p_i by q_i p_i for inherited columns. Keep original Gaussian g in
the newborn geometry, birth selection, trace carry and every other
consumer. In particular, compute innovation and all newborn parameters
from the ORIGINAL inherited plane g_i^2 g_j K21_CP. Change cp_post's
inherited term to q_i^2 q_j K21_CP and restore its repeated-plane
difference in the existing star. No new rank or source strength.

For coherent co-adaptation, the response is explicitly a frozen auxiliary
P0 model: maintain its own recurrent state from current pre-correction
features. It is the bank's learned derivative, NOT the derivative of
subsequently retrained P1. P1 is the main mean correction. This keeps the
response law identical in teacher rows and deployment; no stale-row
claim that response parameters co-adapted. Initially P1=P0 and their
value/state evaluations agree; later P0 still uses the actual current
features, never the current true correction. Its extra forward cost
must be metered if the pilot earns a candidate. No response-only fit.

Prerequisites: synthetic float64 full symmetric third-tensor certificate
for all-distinct change and zero repeated-plane change, relative<=1e-12.
Current rank35/100 Gaussian-response disabled path must be byte-identical
to the original dynamic P/Q trajectory, including final input/state.
On the enabled path record pre-correction response capture and declared
post repeated-plane residuals. Existing float32 factor/star rounding is
reported honestly, not described as exact full tensor preservation.
Unchanged source L0..7 must remain byte-identical in both trajectories.

Then one bounded 20/4 two-stage matched trial, ranks35..58, holdouts
37/39/42/48, identical to the archived positive-path CONTROL trial.
Reuse its unchanged control rows/payloads/receipts; do not retrain an
identical control. Generate exactly24 honest response rows, features and
P0 response fixed before current truth feedback. Inspect L10..15 profile
with the same healthy-profile gate as the control/source prior pilot.
Main prefix P1: same existing GNN trainer, bank initialization, seed
1753362478, data-order seed40,40 epochs, LR5e-4 to5e-5, weight decay2e-5,
same normalized layer loss, final epoch only. No extra features/heads.
Then freeze24 fully on-policy scalar-final rows using P1 mean and fixed
P0 response, no truth state feedback. Fit original scalar Q architecture
and initialization for40 epochs using the exact archived control setup.

Judge all four actual NumPy closed-loop endpoints against existing1e9
truths. Require new/control pooled final MSE<=.90, new/bank<=.92, and
at least3/4 wins versus original bank. Only PASS licenses a separate full
200-net matched round, cost/native parity,14-net both-panel validation,
persistent runtime/memory and eventual submission. Failure closes this
fixed response transport: no layer/strength/epoch/seed/feature sweep.
