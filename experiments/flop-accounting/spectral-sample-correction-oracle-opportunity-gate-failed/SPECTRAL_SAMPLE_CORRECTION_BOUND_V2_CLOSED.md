# Spectral sample correction: oracle opportunity gate failed

2026-09-05, after 12:44 PDT. Phase 2 only. No new samples or fit.

The repaired audit uses the exact scalar-final bank reference (byte-identical
on ranks35/100) and a separate target-blind full-pair endpoint covariance
read of its prefix. Existing documented1e9 means supply the diagnostic error;
coherent8M stored covariance supplies the iid sample-mean variance model.

Even nondeployable oracle per-mode gains fail to pay for sampling at every
predeclared N. At N=8192, predicted raw ratios are0.983697/0.986034 but
optimistic adjusted ratios are1.291249/1.295004, versus each development
net's own bank mean and measured first-context bill. At N=512 adjusted
ratios are1.019069/1.019266; larger counts worsen the tradeoff. Eigensolver,
random generation, projection and reduction cost was omitted, not hidden
inside a claimed measured bill. N=131072 is additionally over budget.

The low-variance modes contain little current mean-error energy. Modes
below0.1 times average output variance hold only6.70%/7.42% of the error;
modes below0.01 hold0.345%/0.290%. No empirical projected variance was
negative. Using the card variance model instead leads to the same verdict.

This closes the tested diagonal spectral shrinkage approach in this basis;
it is not a theorem against every statistical estimator. No sample-count
sweep, calibration fit, sample generation or candidate port is justified.
No official score improvement or upload. Receipts:
spectral_sample_correction_bound_r35_v2.json and
spectral_sample_correction_bound_r100_v2.json. Failed v1 script preserved.
