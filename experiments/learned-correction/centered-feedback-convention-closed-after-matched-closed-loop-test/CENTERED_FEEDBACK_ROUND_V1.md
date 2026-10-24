# Translation-consistent mean feedback: one two-net screen

September5,2026. Current mean feedback replaces the mean while holding raw
second moments fixed: C <- C-m d^T-d m^T-d d^T. Central higher cumulants
are not fully recentered, as documented by the closed third-recenter round.

An alternative coherent moment interpretation is a deterministic shift of
the modeled post activation: Y_new=Y_old+d. Then its mean changes by d and
all centered moments remain unchanged. Test that exact identity as a
different feedback convention: leave C and current central-moment source
calculations unchanged while applying the same mean correction. This is
not the rejected raw-third-moment-preserving recenter, nor the historical
unmatched open-loop operator fed into the old D11 recursion.

One honest teacher-forced test on current full-pair host, ranks35/100.
No old learned head is applied to the changed state. All host/features are
frozen before true mean feedback. Require >=10% reduction in mean one-step
MSE at layers10..15 on BOTH nets versus the current parent's saved honest
rows; finite states and identical layer0. No coefficient sweep or damping.
If it fails, close this exact convention without retraining. If it passes,
matched200 rows/heads and closed-loop/panel/runtime gates remain mandatory.
