certificate on public network 35: N = 64094208 samples of mine against the organizers' ~1e9. VERDICT: PASS
| layer | mean-squared difference (x1e-9) | predicted sampling noise (x1e-9) | plain ratio (chance sd) | ratio to variance / N alone | whitened ratio (chance sd 4.4%) | difference along the mean, standard errors | halves A vs B: plain / whitened | my variance vs the 32M run: mean ratio - 1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 12.047 | 11.320 | 1.064 (5.5%) | 1.132 | 1.011 | -0.29 | 1.017 / 1.022 | -9.9e-06 |
| 1 | 8.867 | 8.432 | 1.052 (6.9%) | 1.119 | 1.016 | -0.11 | 0.984 / 1.019 | +2.0e-05 |
| 2 | 7.043 | 6.626 | 1.063 (8.1%) | 1.131 | 0.985 | +0.96 | 1.009 / 1.028 | +1.2e-05 |
| 3 | 5.669 | 5.512 | 1.029 (9.1%) | 1.094 | 0.982 | +0.84 | 1.089 / 1.002 | +1.5e-06 |
| 4 | 4.759 | 4.758 | 1.000 (10.1%) | 1.064 | 1.040 | +0.85 | 1.095 / 0.992 | +2.3e-04 |
| 5 | 3.645 | 3.914 | 0.931 (11.0%) | 0.991 | 0.964 | +0.89 | 1.140 / 1.004 | +1.2e-04 |
| 6 | 3.400 | 3.435 | 0.990 (11.9%) | 1.053 | 0.993 | +0.64 | 1.149 / 0.964 | +1.4e-04 |
| 7 | 2.699 | 2.897 | 0.932 (12.7%) | 0.991 | 0.984 | +1.05 | 1.231 / 0.983 | -2.0e-04 |
| 8 | 2.559 | 2.602 | 0.983 (13.7%) | 1.046 | 0.986 | +1.04 | 1.260 / 1.002 | +1.8e-04 |
| 9 | 2.225 | 2.129 | 1.045 (14.5%) | 1.112 | 0.984 | +0.91 | 1.224 / 1.054 | +1.9e-04 |
| 10 | 1.859 | 1.867 | 0.996 (15.1%) | 1.059 | 1.012 | +0.97 | 1.385 / 1.007 | -1.1e-04 |
| 11 | 1.661 | 1.733 | 0.958 (16.6%) | 1.020 | 0.985 | +0.86 | 1.467 / 0.964 | +6.3e-05 |
| 12 | 1.578 | 1.567 | 1.007 (16.8%) | 1.072 | 0.981 | +1.04 | 1.623 / 1.064 | +7.0e-05 |
| 13 | 1.279 | 1.447 | 0.884 (18.2%) | 0.941 | 1.010 | +1.10 | 1.496 / 1.018 | +1.0e-04 |
| 14 | 1.236 | 1.276 | 0.968 (18.7%) | 1.030 | 0.989 | +1.28 | 1.618 / 1.035 | +1.4e-04 |
| 15 | 1.179 | 1.118 | 1.055 (19.6%) | 1.123 | 1.019 | +1.25 | 1.662 / 0.973 | +2.4e-04 |
whitened ratio: mean over layers 0.996, range 0.964-1.040; layers more than 10% off: none; plain ratio within 10% in 15 of 16 layers
generator checks: mc_truth.draw, 1.3e+10 values: variance - 1 = +1.3e-05 (+1.0 se), excess kurtosis -1.4e-04, largest |x| 5.17; mc_truth.draw, 1e+11 values: variance - 1 = -2.0e-06 (-0.4 se), excess kurtosis -6.2e-05, largest |x| 5.17; mc_truth.draw, 1e+11 values: variance - 1 = -9.1e-07 (-0.2 se), excess kurtosis -1.1e-04, largest |x| 5.17; raw torch.randn on MPS, 1.3e+10 values: variance - 1 = +3.8e-05 (+3.1 se), excess kurtosis +4.9e-03, largest |x| 14.37; mc_truth.draw, 1.3e+10 values: variance - 1 = +1.7e-07 (+0.0 se), excess kurtosis -7.9e-05, largest |x| 5.17
gain-direction test rng_gain_test_seed5550001 (1000 batches of 32768): layer 3: batch-mean variance x batch / lambda1 = 1.090, lag-1 -0.016, mean vs organizers +0.50 se; layer 8: batch-mean variance x batch / lambda1 = 1.023, lag-1 -0.010, mean vs organizers +0.34 se; layer 15: batch-mean variance x batch / lambda1 = 1.075, lag-1 -0.002, mean vs organizers +0.80 se
gain-direction test rng_gain_test_seedB (1000 batches of 32768): layer 3: batch-mean variance x batch / lambda1 = 1.065, lag-1 -0.054, mean vs organizers +2.35 se; layer 8: batch-mean variance x batch / lambda1 = 1.022, lag-1 +0.009, mean vs organizers +2.47 se; layer 15: batch-mean variance x batch / lambda1 = 1.039, lag-1 +0.027, mean vs organizers +2.71 se
