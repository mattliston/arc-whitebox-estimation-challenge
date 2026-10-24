# Classical covariance response: final qualification failed

Changing only covariance transport to ordinary FP32 matrix products at
layers9..15 reduces the post9 output secant gaps from61.30/62.55% to
2.9905/2.9495%, passing all post9 gates on MINI57/68. The full-estimator final
gaps are45.52/48.33%, with curvature0.3818/0.3897. Both final cells fail the
fixed5% secant and0.10 curvature gates. Direction7 is therefore not run.

| MINI | Post-layer | Relative secant gap | Curvature | RMS(h*r_half) |
|---:|---:|---:|---:|---:|
| 57 | 9 | 0.029905 | 0.023215 | 9.4097012e-07 |
| 57 | 10 | 0.135130 | 0.097577 | 1.3985745e-06 |
| 57 | 11 | 0.139167 | 0.099466 | 1.9155625e-06 |
| 57 | 12 | 0.153568 | 0.110055 | 2.1909331e-06 |
| 57 | 13 | 0.349642 | 0.265295 | 2.6461977e-06 |
| 57 | 14 | 0.423379 | 0.342690 | 2.8645468e-06 |
| 57 | 15 | 0.455155 | 0.381828 | 3.2301842e-06 |
| 68 | 9 | 0.029495 | 0.024857 | 1.0116274e-06 |
| 68 | 10 | 0.117362 | 0.095967 | 1.5933254e-06 |
| 68 | 11 | 0.139378 | 0.107455 | 1.9843303e-06 |
| 68 | 12 | 0.150229 | 0.117349 | 2.3036121e-06 |
| 68 | 13 | 0.350939 | 0.274750 | 2.6316735e-06 |
| 68 | 14 | 0.429892 | 0.333979 | 2.9438523e-06 |
| 68 | 15 | 0.483266 | 0.389711 | 3.178038e-06 |

The same fixed direction0 and signed steps1/8,1/16 are used. This is a new
covariance-arithmetic intervention, not a rescue or rescore of checkpoint48.
No final reference means are read, no probe MSEs are scored, and no output
coefficients are fit. The experimental source is not admitted for deployment.
The first remaining increase is at layer10; another large increase occurs at
layer13. Those are current shared-old join layers, which motivates inspection
but does not by itself prove that a join is the remaining cause.

Disabled replacement reproduces all16 outputs and complete bills. Enabled
calls preserve first9 outputs, make exactly six full covariance replacements
and one terminal diagonal replacement, and retain finite nonnegative means.
At layer9, captured C/star inputs and star output remain byte-exact; the new
covariance output equals an independent ordinary FP32 replay byte for byte.
Undoing the two call substitutions and removing the declared footer recovers
the original supplier exactly. Independent helper AST checks, readout/prefix
checks and scalar fsum response replay pass.

The replacement increases the already helper-bearing bill by4.47093/4.38081%.
No accuracy/economic gain follows. All8 native-study guards complete and
retire; all16 combined guards peak at5.245GB with zero added swap.
The live controller is reaped. Old covariance and shared-sketch precision
failures remain preserved; no parameter, radius, scope or precision tuning
follows within this experiment.

S31#332517 remains the saved official bank at2.1736465056908115e-9 adjusted.
No fresh poll, approval or grading is pending. Submissions remain authorized.
The2e-9/~1e-9 goal remains active and unmet.
