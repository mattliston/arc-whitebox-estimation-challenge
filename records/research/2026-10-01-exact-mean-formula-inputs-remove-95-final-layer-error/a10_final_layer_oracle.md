
Final-layer MSE x1e-9 with the inputs of the mean formula made exact AT LAYER 15 ONLY, learned correction ON (change against the torch card as it is):
| inputs made exact at layer 15 | network 35 | network 100 | network 37 | network 39 | mean of four |
|---|---|---|---|---|---|
| none (the card as it is) | 14.11 (+0%) | 15.37 (+0%) | 16.26 (+0%) | 14.26 (+0%) | 15.00 (+0.0%) |
| variance | 13.17 (-7%) | 13.21 (-14%) | 15.51 (-5%) | 13.02 (-9%) | 13.73 (-8.5%) |
| skew | 13.89 (-2%) | 15.09 (-2%) | 15.83 (-3%) | 13.86 (-3%) | 14.66 (-2.2%) |
| kurtosis | 13.33 (-6%) | 14.80 (-4%) | 15.33 (-6%) | 13.92 (-2%) | 14.34 (-4.4%) |
| variance + skew | 12.75 (-10%) | 12.83 (-17%) | 15.01 (-8%) | 12.19 (-14%) | 13.20 (-12.0%) |
| variance + kurtosis | 11.86 (-16%) | 12.43 (-19%) | 14.32 (-12%) | 12.56 (-12%) | 12.79 (-14.7%) |
| skew + kurtosis | 12.82 (-9%) | 14.30 (-7%) | 14.70 (-10%) | 13.34 (-6%) | 13.79 (-8.1%) |
| all three | 11.16 (-21%) | 11.82 (-23%) | 13.62 (-16%) | 11.55 (-19%) | 12.04 (-19.7%) |

Final-layer MSE x1e-9 with the inputs of the mean formula made exact AT LAYER 15 ONLY, learned correction OFF at layer 15 only (change against the torch card as it is):
| inputs made exact at layer 15 | network 35 | network 100 | network 37 | network 39 | mean of four |
|---|---|---|---|---|---|
| none (the card as it is) | 15.03 (+7%) | 15.94 (+4%) | 17.02 (+5%) | 15.06 (+6%) | 15.76 (+5.1%) |
| variance | 14.00 (-1%) | 13.73 (-11%) | 16.06 (-1%) | 13.82 (-3%) | 14.40 (-4.0%) |
| skew | 14.22 (+1%) | 15.02 (-2%) | 15.91 (-2%) | 14.00 (-2%) | 14.79 (-1.4%) |
| kurtosis | 13.97 (-1%) | 14.92 (-3%) | 15.79 (-3%) | 14.50 (+2%) | 14.79 (-1.4%) |
| variance + skew | 13.00 (-8%) | 12.71 (-17%) | 14.89 (-8%) | 12.34 (-13%) | 13.23 (-11.8%) |
| variance + kurtosis | 12.41 (-12%) | 12.50 (-19%) | 14.58 (-10%) | 13.13 (-8%) | 13.16 (-12.3%) |
| skew + kurtosis | 12.87 (-9%) | 13.78 (-10%) | 14.49 (-11%) | 13.26 (-7%) | 13.60 (-9.3%) |
| all three | 11.13 (-21%) | 11.25 (-27%) | 13.21 (-19%) | 11.46 (-20%) | 11.76 (-21.6%) |
