Base torch card, final-layer error against the mean per-unit variance of layer 15 (census: 660 public, 300 generated networks)

| population | networks | mean error (1e-9) | log error on log variance: slope (se) | correlation | sd of log error | residual sd | sd of log variance |
|---|---|---|---|---|---|---|---|
| public, old TRAIN160 (trained on) | 96 | 16.56 | +0.53 (0.07) | +0.61 | 0.122 | 0.097 | 0.139 |
| public, validation 40 (selected on) | 40 | 16.96 | +0.57 (0.17) | +0.49 | 0.135 | 0.118 | 0.115 |
| public, fresh (neither) | 304 | 17.25 | +0.55 (0.05) | +0.54 | 0.133 | 0.112 | 0.131 |
| public, all with a base error | 440 | 17.07 | +0.55 (0.04) | +0.55 | 0.132 | 0.110 | 0.132 |
| generated | 35 | 17.64 | +0.85 (0.16) | +0.68 | 0.184 | 0.135 | 0.147 |

Slope, generated minus fresh public: +0.30 (se 0.17, z +1.8). Residual spread, generated against fresh public: 0.135 against 0.112, Brown-Forsythe p 0.130.
Layer-15 variance: public 660: mean 0.0752, sd / mean 13.6%; generated 300: mean 0.0756, sd / mean 13.1%; p(means) 0.60, p(spreads) 0.77, p(KS) 0.48.
Layer-3 variance: public mean 0.3199, sd / mean 4.3%; generated mean 0.3215, sd / mean 4.1%; p(means) 0.08, p(spreads) 0.65.
Layer-7 variance: public mean 0.1680, sd / mean 7.8%; generated mean 0.1691, sd / mean 7.7%; p(means) 0.23, p(spreads) 0.61.
Layer-11 variance: public mean 0.1066, sd / mean 11.0%; generated mean 0.1071, sd / mean 10.7%; p(means) 0.58, p(spreads) 0.69.
Layer-13 variance: public mean 0.0888, sd / mean 12.2%; generated mean 0.0887, sd / mean 12.1%; p(means) 0.96, p(spreads) 0.52.
