# P2R0G1 Kerdock h7 anchor ceiling

This is a one-shot oracle discriminator for the internal Kerdock re-anchoring
route.  It does not sample network outputs and it is not deployable: it asks
whether one perfect residual-state reset at post-ReLU `h_7` would be sufficient
to carry the already frozen 6,688-parameter recurrence below raw MSE `1e-8`.

The anchor was fixed from the independent compute/consumer contract before the
result was computed: `N=65,536` structured points can reach `h_7`, and the
partition-complete cumulant lane requested its projected `h_7` state.  After the
single exact reset, layers 8 through 16 are strictly autonomous.  No alternate
layer is evaluated.

Only the already opened public `observer_fit` development-validation ranks
160:199 are read.  `observer_selection`, names, board feedback, deployment
truth, and `observer_fit` targets at rank 200 or later are untouched.
