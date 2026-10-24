# Cheap pair-kernel prediction of fresh covariance error: fixed pilot

September5,2026, target now below1e-9. Existing current-host covariance
geometry shows most accumulated error has at least one high-variance
direction, but it does not supply a deployable correction. Prior fixed/
learned graph MEAN heads were unhelpful. The old Hadamard-kernel negative
was end-to-end training on a Gaussian DC1 host with raw1.17e-6, not this
supervised source test on the much more accurate current moment carrier.

First test whether a small response-basis pair kernel predicts FRESH
covariance-map error on unseen networks. Eight existing8M covariance
truths are available; no new MC or private targets are needed. Use six
train nets35/100/150/200/250/300 and held-out400/500; layers8..14 only.

For each trajectory force past means to stored1e9 means and past covariance
to stored8M post covariance. Keep the full current third/radial carrier.
At each layer capture the nonlinear host C, standardized pre-correlation,
and14 local response basis functions BEFORE current truth feedback.
The target is true post covariance minus this nonlinear host covariance.
Truth covariance is then supplied for future steps and trace birth state;
restore its full matrix after compact masking for the next teacher input.
This is an explicitly oracle STATE distribution for supervised information
diagnosis, NOT a deployable candidate or a measured closed-loop score.

Predict E_ij = sigma_i sigma_j sum_{r=0..2}
R_ij^r * (F_i^T Theta_r F_j), with symmetric14x14 Theta_r shared across
layers. Basis: gate, pdf, alpha*pdf, (alpha^2-1)*pdf, host/sigma, alpha*gate,
and skew/kurtosis times gate/pdf/alpha*pdf/host/sigma. Normalize basis RMS
on TRAIN nets only. This is315 independent linear coefficients, no hidden
neural capacity or basis/rank sweep. Public-operation deployment cost is
matrix products of thin14-channel factors plus Hadamard products.

One fixed100-epoch Adam fit, learning rate.02 decaying to.002, ridge1e-3.
Objective balances covariance Frobenius MSE and the next-layer Gaussian
mean response to diag(W_next^T E W_next); normalize each by its train
zero-predictor energy. Inspect sampling uncertainty, but do not claim a
noise-corrected representation ceiling from non-independent cov/cov2.

At the final epoch require >=25% reduction of BOTH covariance-error MSE
and next-mean-response error on EACH held-out net400/500. Otherwise close
this fixed kernel without more covariance labels,200-net training, or port.
If it passes, positivity, fresh-noise validation, student-state exposure,
matched mean heads, actual14-net raw/adjusted and runtime gates still remain.
