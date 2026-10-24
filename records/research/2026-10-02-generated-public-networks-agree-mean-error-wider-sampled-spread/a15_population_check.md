## A15. Generated networks against public networks as judges of the base card (35 generated seeds)

Final-layer MSE of the current torch card with nothing added, per network, in units of 1e-9; truth noise subtracted where it is known.

| population | networks | mean | sd across networks | standard error of the mean | median | range | sd / mean |
|---|---|---|---|---|---|---|---|
| generated, as measured (64M-sample truth, noise included) | 35 | 18.85 | 3.35 | 0.57 | 18.01 | 13.17 to 25.55 | 18% |
| generated, truth noise subtracted (noise 1.21 per network; from the two halves 1.22) | 35 | 17.64 | 3.23 | 0.55 | 16.90 | 12.22 to 24.23 | 18% |
| public, the trainer's 100 held-out networks (organizers' noise 0.077 subtracted) | 100 | 17.21 | 1.96 | 0.20 | 17.00 | 12.51 to 22.36 | 11% |
| public, the lead's 24 of those 100 | 24 | 16.94 | 1.78 | 0.36 | 16.62 | 14.07 to 21.40 | 10% |
| public, every fresh network of the trainer's captures (not old training, not validation): THE REFERENCE | 324 | 17.16 | 2.28 | 0.13 | 16.97 | 10.84 to 24.73 | 13% |
| board: our entry #333111 (native), 50 public board networks, as graded | 50 | 17.12 | 2.13 | 0.30 | 17.02 | 13.91 to 24.10 | 12% |
| the same in torch units (divided by the native / torch factor 1.0120) | 50 | 16.92 | 2.11 | 0.30 | 16.82 | 13.75 to 23.81 | 12% |

Native / torch factor: 1.0120 at the final layer, 1.0101 for the 16-layer average (native bank against the torch card on the same 40 validation networks; per-network sd of the log ratio 1.6%). The lead's 24 logs and the trainer's files agree on the 24 common networks to 0.039%.

| comparison (first minus second) | difference of means | its standard error | z | ratio of means | p, means (Welch) | ratio of sds | p, spreads (Brown-Forsythe) | p, distributions (Kolmogorov-Smirnov) | p (Mann-Whitney) |
|---|---|---|---|---|---|---|---|---|---|
| generated (35) against the 324 fresh public networks, final layer | +0.48 | 0.56 | +0.86 | 1.028 | 0.394 | 1.41 | 0.005 | 0.300 | 0.521 |
| generated (35) against public 100, final layer | +0.43 | 0.58 | +0.74 | 1.025 | 0.464 | 1.65 | 0.001 | 0.104 | 0.664 |
| generated (35) against the lead's public 24, final layer | +0.70 | 0.65 | +1.07 | 1.041 | 0.290 | 1.82 | 0.010 | 0.149 | 0.409 |
| generated against public 100, 16-layer average | +0.18 | 0.16 | +1.16 | 1.028 | 0.252 | 1.59 | 0.003 | 0.456 | 0.279 |
| board (torch units) against public 100, final layer | -0.37 | 0.36 | -1.05 | 0.978 | 0.298 | 1.08 | 0.761 | 0.281 | 0.186 |
| board (torch units) against generated, final layer | -0.80 | 0.62 | -1.29 | 0.955 | 0.203 | 0.65 | 0.011 | 0.123 | 0.259 |
| board (torch units) against public 100, 16-layer average | -0.09 | 0.09 | -1.00 | 0.986 | 0.321 | 0.96 | 0.799 | 0.518 | 0.292 |
| board (torch units) against generated, 16-layer average | -0.28 | 0.17 | -1.66 | 0.961 | 0.102 | 0.60 | 0.009 | 0.108 | 0.113 |

Spread by resampling: samples of 35 of the 324 fresh public networks (with 0.86e-9 of truth noise added, the certificate's figure) reach the generated sd of 3.23 in 0.13% of draws; for the generated seeds 8 and later (27 networks, sd 3.11), which did not prompt the question, 1.07%.

Per-layer error, noise subtracted (mean over networks, 1e-9):

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| generated | 0.12 | 0.21 | 0.56 | 1.03 | 1.77 | 2.77 | 3.72 | 4.96 | 6.26 | 7.47 | 9.01 | 10.66 | 12.30 | 14.10 | 15.52 | 17.64 |
| its standard error | 0.10 | 0.09 | 0.09 | 0.05 | 0.06 | 0.07 | 0.08 | 0.12 | 0.15 | 0.19 | 0.21 | 0.24 | 0.32 | 0.36 | 0.45 | 0.55 |
| public 100 | -0.00 | 0.12 | 0.46 | 1.00 | 1.74 | 2.61 | 3.65 | 4.79 | 6.02 | 7.27 | 8.84 | 10.38 | 11.98 | 13.77 | 15.33 | 17.21 |
| its standard error | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 | 0.04 | 0.06 | 0.07 | 0.09 | 0.11 | 0.13 | 0.16 | 0.17 | 0.20 |
| generated / public | -1193.50 | 1.67 | 1.20 | 1.03 | 1.02 | 1.06 | 1.02 | 1.04 | 1.04 | 1.03 | 1.02 | 1.03 | 1.03 | 1.02 | 1.01 | 1.02 |
| z of the difference | +1.2 | +1.0 | +1.1 | +0.6 | +0.6 | +2.3 | +0.8 | +1.4 | +1.6 | +1.0 | +0.7 | +1.1 | +0.9 | +0.8 | +0.4 | +0.7 |
| truth noise subtracted from generated | 10.64 | 7.95 | 6.25 | 5.08 | 4.20 | 3.56 | 3.06 | 2.65 | 2.33 | 2.08 | 1.87 | 1.70 | 1.54 | 1.42 | 1.30 | 1.21 |
| the same noise measured from the two halves | 10.79 | 8.03 | 6.29 | 5.06 | 4.12 | 3.52 | 3.02 | 2.61 | 2.29 | 2.03 | 1.84 | 1.67 | 1.55 | 1.44 | 1.31 | 1.22 |

Activation scale (mean square of the true final-layer means): generated 0.9653 (sd 0.2484), public 100 0.9235 (sd 0.2243), all 688 public 0.9202 (sd 0.2208); generated against all 688: p(means) 0.30, p(spreads) 0.51, p(KS) 0.52. Log error against log scale: slope 0.05 (correlation 0.12) on the public 100, 0.20 (0.26) on the generated. Error divided by scale, generated against public 100: ratio of means 0.973, p(means) 0.589, p(spreads) 0.918.

Spread of the generated networks' measured error that is truth noise (two-halves estimate, 35 networks): final layer: cross term sd 0.58, noise-square term sd 0.29, together 0.65 (its own uncertainty about 12%); measured sd 3.23; so the network-to-network sd of the true error is 3.16, against 1.96 on the public 100.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| generated: sd of the measured error | 0.61 | 0.50 | 0.53 | 0.32 | 0.35 | 0.39 | 0.48 | 0.69 | 0.87 | 1.12 | 1.23 | 1.42 | 1.90 | 2.13 | 2.65 | 3.23 |
| of which truth noise (sd) | 0.63 | 0.58 | 0.61 | 0.50 | 0.45 | 0.40 | 0.39 | 0.40 | 0.53 | 0.54 | 0.52 | 0.59 | 0.62 | 0.67 | 0.73 | 0.65 |
| generated: sd of the true error | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.27 | 0.56 | 0.68 | 0.98 | 1.11 | 1.29 | 1.79 | 2.02 | 2.55 | 3.16 |
| public 100: sd | 0.04 | 0.06 | 0.07 | 0.09 | 0.14 | 0.21 | 0.28 | 0.38 | 0.55 | 0.68 | 0.89 | 1.13 | 1.32 | 1.64 | 1.71 | 1.96 |

Per generated network (1e-9): 0: 22.5, 1: 13.7, 2: 24.2, 3: 16.9, 4: 19.4, 5: 20.7, 6: 15.8, 7: 16.7, 8: 19.0, 9: 12.3, 10: 12.6, 11: 15.8, 12: 15.7, 13: 17.8, 14: 15.7, 15: 23.3, 16: 16.8, 17: 17.6, 18: 19.0, 19: 20.3, 20: 18.2, 21: 19.0, 22: 13.6, 23: 16.0, 24: 20.6, 25: 15.7, 26: 21.9, 27: 16.9, 28: 14.8, 29: 16.5, 30: 14.7, 31: 20.7, 32: 12.2, 33: 17.1, 34: 23.6
