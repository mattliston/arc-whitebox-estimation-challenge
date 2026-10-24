# Adaptive Gaussian mixture literature/code audit

September6,2026, Phase2. No code transplant, network sampling or fit.

Read primary PMLR abstract and authors' public implementation:
https://proceedings.mlr.press/v202/monchot23a.html
https://github.com/PaulMcht/WGMprop
https://github.com/PaulMcht/WGMprop/blob/master/model/utils_wgmprop_propagation.py
https://github.com/PaulMcht/WGMprop/blob/master/run_param_mnist.json

The released full-network propagation uses2q+1 unscented points per
Gaussian component in a q-dimensional covariance subspace, evaluates both
the network and a linearization, and splits components using a Wasserstein-
based criterion. The example permits500 components, with subspace disabled.
For this task's full-rank1024-dimensional Gaussian, one such component
therefore uses2049 full-network points before other work. This is an input
quadrature route, not a demonstrated inexpensive analytic conditional
moment recursion. No published1024x16/1e-9 mean-MSE or metered-cost result
was identified in these sources. A naive transplant has no demonstrated
advantage over the already unsuccessful input rules; do not run it merely
because it is called a Gaussian mixture.

This is NOT an impossibility result for all mixtures. Full paper PDF was
not read in this audit. The prior Zhang/Shin2021 publisher abstract and
introduction concern adaptive splitting/reduction and likewise do not
establish challenge-scale precision or cost.
https://doi.org/10.1016/j.neucom.2021.06.007

Separate arithmetic checks: the bank already exploits exact sharing of
the newest signed pair's first weight push (build_shared_newborn_push_v1.py).
The old signed-pair cap scores the joint cancelling pair energy, not two
separate cube energies (radial_binary_cap.py). Neither is a newly found
compression bug. Float16 and float32 have the same installed FlopScope
rate; this was already documented. No duplicate compute was launched.
