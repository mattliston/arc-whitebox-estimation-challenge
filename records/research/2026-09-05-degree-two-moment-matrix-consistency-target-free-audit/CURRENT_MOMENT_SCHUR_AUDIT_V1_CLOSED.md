# Joint moment-PSD repair is not indicated

September 5, 2026. Target-free current-parent pre-state audit on ranks
35/100, layers 4/8/12/15, restricted to the actual active blocks. All eight
covariance systems passed the declared conditioning test; no regularization
or state repair was performed. Every Schur complement is strictly positive.
Smallest eigenvalues range from 0.6017 to 1.3952 in standardized units.
There are zero negative eigenvalues and zero scalar Cauchy violations.

At layer 15, the largest scalar regression-energy / squared-variable-
variance ratios are .19936 / .18776, well below one. Thus this necessary
moment-cone constraint has substantial slack. No PSD correction strength,
model fit or candidate port is justified by this audit. It says nothing
about whether the admissible K3/K4 values are accurate.

Receipt: current_moment_schur_audit_v1.json. No labels were read.
