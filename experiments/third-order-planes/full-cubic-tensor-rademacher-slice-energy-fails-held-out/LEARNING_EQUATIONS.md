# Frozen source regression equations

For layer l in8..15, let s,alpha,pdf be the parent inputs and F its original
FP32 features. Responses are y3=k3_reference/s^3-F10 and
y4=k4_reference/s^4-F12. The references are existing coherent8M moment
estimates. Weights are w3=s*(-alpha*pdf/6) and
w4=s*((alpha^2-1)*pdf/24). The objective is the SUM of the two mean squared
weighted cumulant residuals, not their squared sum and not final-layer loss.
This does not model all interactions with the parent's learned correction.

TRAIN q is centered and scaled by layer across all six networks and1024
coordinates. Third controls are16 columns: alpha and F10 separately for
each layer. Fourth controls are8 layer indicators. Extended models add two
global third columns q_normalized*alpha and q_normalized*F10, and one global
fourth column q_normalized. No clipping, nonlinear transform, alternate
probe seed, or replacement feature selection occurs.

For each of the four fits, M=diag(w)X and t=w*y. Scale each column of M by
its TRAIN RMS, with numerical floor max(max(column_RMS)*1e-12,1e-30).
With Mbar the scaled matrix and N=49152 rows, solve
(Mbar.T Mbar/N + .001 I) theta = Mbar.T t/N once inFP64. Convert theta back
to coefficients on X. All coefficients, including layer indicators, receive
that same penalty in these scaled coordinates. Verify the normal equation
residual <=1e-10. The24-parameter control and27-parameter extension are both
fitted once. Reload the saved coefficients before opening CHECK arrays.

Report each third/fourth component, every layer, each network and pooled
TRAIN/CHECK. Primary gates remain the original protocol's CHECK <=.85
versus zero prediction and <=.95 versus fitted control, with each network
<=1 against both. CHECK networks35/100 have been used extensively before;
there are only two of them, and these reference losses have sampling error.
No confidence interval, population bound, physical cumulant closure,
native score gain, or official-score projection follows from this screen.

The source labels do not use final1B answers, but they are privileged TRAIN
supervision. At inference the proposed feature depends only on the network
and current parent state. This screen is the sole planned fit for this
descriptor. Failure leaves the feature closed without a native candidate.
