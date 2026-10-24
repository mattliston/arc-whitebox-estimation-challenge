# Current-carrier transport shadow: fidelity gate before any fit

September 5, 2026. Goal: adjusted below 1e-9; current official bank 4.982e-9.
The recent source pair-kernel pilot is closed. Prior teacher-forced mean
heads, D11 scheduled sampling, and Gaussian-card end-to-end fits are not
evidence for training through the much more accurate current carrier.

Test a defect-corrected differentiable shadow of the current full-pair
learned trajectory. Propagate perturbations of means and full covariances
through the actual weight matrices and analytic nonlinear mean/covariance
map. Use the exact current baseline pre-K21 and pre-K4 as frozen inputs;
their perturbations are omitted, as are active-mask derivatives. Recompute
the node features and recurrent GNN on the changed state. Subtract each
shadow baseline evaluation, so the original payload is an exact fixed
point (up to the stored output dtype). No labels enter this construction.

This is an approximate gradient model, not a candidate. It may fail because
K3/K4 state responses matter or mask changes dominate. Before fitting,
compare its final response with central perturbations of the real NumPy
current carrier, original head active, ranks 35/100. Three fixed random
per-neuron correction directions inserted at layers 4/9/13, each unit-RMS,
epsilon 2e-4. Also use half epsilon to check finite-difference stability.
Directions are generated without reading any target. Fixed seed per rank
and layer. Require relative L2 response error <=10% for every direction,
and parent output agreement <=3e-7. Verify an autograd adjoint against the
shadow's finite difference. No fitting is authorized by this protocol
unless those numerical gates pass; a passing shadow is not a raw gain.

If passed, design one bounded warm-start end-to-end pilot separately, with
fixed public train/validation split and an actual full-carrier held-out
gate. No target-fed features, sealed target access, production edits,
forum writes, or submissions occur in this fidelity experiment.
