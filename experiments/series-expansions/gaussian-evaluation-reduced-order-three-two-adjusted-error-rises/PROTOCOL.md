# Unselected Gaussian R2 component proposal

Status: source-only draft, not admitted for execution. Prospective parent is
88683ee1, irrespective of its pending release outcome. No VALID40 outcome was
read to prepare this proposal. This revision prepares one source-only candidate and primitive/config draft; no numerical job is authorized.

One fixed change: Gaussian series 3:2:2 to 2:2:2 at the existing recurrent pair
calls. Three literal edits shorten Fr16/Gr16/P16/P6 from [1:4] to [1:3], DIV6
from arange(1,4) to arange(1,3), and X1 from (3,3,n,n) to (2,3,n,n). A comment
changes R3 to R2. All derivatives0..6, RR/P12, E2/EL2, scalar moments, learned
coefficients, dtypes, first Gaussian layer and terminal scalar path are retained.
The P4[128,128] symmetry partition, all setup keys and original arena are fixed.

R2 is the lowest order that still recomputes both powers consumed by RR/P12.
R1 with these unchanged corrections would leave a required rho-squared plane
stale and is explicitly excluded. R2 removes a genuine Gaussian cubic term;
this is an approximation, not a byte-exact optimization. Higher cumulant
corrections remain intact. An accuracy or branch-cost benefit is not promised.

Suggested future component gates, requiring separate root admission before any
job: fixed TRAIN2(35,100) must have finite outputs, exact L0 versus88683,
strictly lower pooled adjusted score and two wins against88683. Only then may
the fixed TRAIN6(36,45,52,58,64,70) extend it. Combined TRAIN8 must have at least .03%
lower adjusted score against88683, at least6/8 wins and negative upper95%CI
(50000 resamples, seed20260926). There is no additional-six gain gate. Raw error is reported without a separate
veto. These are proposals, not current execution authorization.

The component saving is about .06%, far below a standalone1% bank claim.
Any eventual bank must independently clear the standing >=1% fully validated
gain against the official comparator fixed at that future admission, all
VALID40 gates and complete serving qualification. Keep105/4.1/.35/7.5GB,
zero-added-swap and70000-handle limits. No order sweep, correction reduction,
layer scope change, refit, coefficient adjustment or post-outcome rescue.

`build.py` performs only the authorized source preparation and exact-inverse checks.
The source-only primitive and inactive native config require separate root execution admission.
