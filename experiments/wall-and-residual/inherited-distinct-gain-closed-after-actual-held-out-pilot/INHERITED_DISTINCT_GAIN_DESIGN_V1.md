# Learned inherited-distinct transport, fixed one-direction pilot

Phase2 only, September5,2026,22:14PDT. User goal<1e-9; all release gates
and no-forum rule unchanged. This is not a birth-amplitude fit, a marginal
k3 rescale, a cutoff/age compression, or an encoder/head refit.

At each post layer1..14, replace the inherited CP factors g*pushed_p by
g*(1+theta_l)*pushed_p. The inherited CP tensor thus gains(1+theta_l)^3.
Keep birth construction based on the ORIGINAL inherited plane. Adjust
only the CP-post plane used in the existing coordinate-star restoration,
so the total post K21/repeated moments remain unchanged. All mean/variance,
trace tensor, active/birth choices, P and finalQ heads stay as currently
defined. No tensor column is removed or added. The changed state action
is exactly a gain on the CP inherited all-distinct component, not a new
physical claim about the Gaussian gate. Index0 is fixed at zero.

Target-free prerequisite on ranks35/100: zero-control NumPy replay must
be byte-identical to current parent; CPU Torch must agree within2e-6.
Use fixed theta_l=.003*sin(2*pi*l/15), then half strength. Require Torch
versus NumPy final-action relative discrepancy<=.10 and positive-step
halving discrepancy<=.05 on both nets. No relaxed tolerance or fit if
this fails; a failing positive-step halving is not an accuracy result.

After fidelity only: one end-to-end train-gradient direction on
35/36/38/40, final-MSE/2e-8 loss, frozen P and Q. Normalize direction by
its maximum absolute entry, with index0 zero. Set its scalar from the
train-only exact forward-AD least-squares response, clipped to[0,.02].
Compare reverse/forward directional derivatives at1% tolerance. No
optimizer, multiple directions, radius/step/epoch/rank/layer selection.
Evaluate the one finite step on pilot holdouts37/39/42/48 using actual
dynamic NumPy and independently replay CPU Torch. Original1e9 labels.
Require pooled final raw ratio<=.90 and3/4 wins before any expansion.

A pass would still need complete metered cost, matched14-net panel and
persistent total/residual-wall plus memory certification. Small physical
gain parameters do not imply negligible measured overhead. No score claim
from a tangent or teacher-forced loss. One nice10 CPU worker,3 threads;
AP Torch never imports threadpoolctl. Preserve every version and output.
