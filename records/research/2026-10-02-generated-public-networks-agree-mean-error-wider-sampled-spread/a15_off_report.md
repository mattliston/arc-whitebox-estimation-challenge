Learned mean correction off against on, final layer, 1e-9 (35 generated, 35 public held-out networks)

| population | networks | off: mean error | sd of log | on: mean error | sd of log | pooled effect of the learned correction | sd of its log effect across networks | log error on log layer-15 variance, off: slope (se) | on: slope (se) | log effect on log variance: slope (se) | correlation of the log effect with the log error off |
|---|---|---|---|---|---|---|---|---|---|---|---|
| generated | 35 | 20.26 | 0.192 | 17.64 | 0.184 | -12.9% | 0.056 | +0.93 (0.16) | +0.85 (0.16) | -0.08 (0.06) | -0.30 |
| public held-out | 35 | 19.67 | 0.154 | 17.15 | 0.139 | -12.8% | 0.048 | +0.69 (0.14) | +0.57 (0.14) | -0.11 (0.06) | -0.44 |

log error, learned off: sd generated 0.192 against public 0.154, ratio 1.25, Brown-Forsythe p 0.206; means differ by +0.023 (p 0.57)

log error, learned on: sd generated 0.184 against public 0.139, ratio 1.32, Brown-Forsythe p 0.081; means differ by +0.022 (p 0.58)

log effect of the learned correction: sd generated 0.056 against public 0.048, ratio 1.16, Brown-Forsythe p 0.341; means differ by -0.002 (p 0.89)

Per layer, sd of log error (learned off): L7: gen 0.128 / pub 0.084 | L9: gen 0.153 / pub 0.093 | L11: gen 0.127 / pub 0.144 | L13: gen 0.157 / pub 0.168 | L14: gen 0.170 / pub 0.150 | L15: gen 0.192 / pub 0.154
Per layer, sd of log error (learned on):  L7: gen 0.137 / pub 0.080 | L9: gen 0.154 / pub 0.090 | L11: gen 0.135 / pub 0.137 | L13: gen 0.149 / pub 0.148 | L14: gen 0.168 / pub 0.137 | L15: gen 0.184 / pub 0.139
