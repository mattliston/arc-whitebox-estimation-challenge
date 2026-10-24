# Frozen birth-direction action — independent14 PASS

Phase2 only. Completed2026-09-08T23:32:43.033424+00:00. This uses the
reused public8/virgin6 independent4M Monte Carlo reference panel, with
unclipped MC-noise subtraction. It is not an official result or pristine
14-model holdout: public35/100/200 were TRAIN and150 was VALID.

| Group | Current409f NC raw | Terminal NC raw | Current409f NC adjusted | Terminal NC adjusted | Adjusted ratio |
|---|---:|---:|---:|---:|---:|
| public8 | 2.43238993902554e-8 | 2.29783886896747e-8 | 5.1858949910628045e-9 | 4.9147243456412085e-9 | .9477099621398192 |
| virgin6 | 1.9971264872362867e-8 | 1.8436635176267465e-8 | 4.2202955997622905e-9 | 3.909232150253503e-9 | .9262934450548184 |
| combined14 | 2.2458484596872894e-8 | 2.103192289821446e-8 | 4.772066680505441e-9 | 4.4837991190464774e-9 | .9395927214016986 |

Each adjusted figure is the mean per-net
(observed final MSE - mean(mc_se15**2)) * max(.1, own complete FLOPs/2^41).
The actual observed combined14 raw means are4.074936483190734e-8 and
3.93228031332489e-8; mean subtracted MC variance is1.8290880235034444e-8.
No clipping or product-of-means score was used.

Both roles were freshly acquired on official-owned FNP weights and paid
their own complete participant bills. Combined14 mean FLOPs:
bank466238038163.9286; terminal467682934475.3571. Initialization is paid
once (1383913666/1383913786 F respectively); repeat35 is byte exact and
excluded. All bank outputs and raw scores reproduce the archived bank
exactly. The new bank bills are actually179306496 F above the older
NumPy-input harness bills per call, not estimated bills used for scoring.

All15 calls per role pass8GB/120s, with peak process RSS6.693GB/6.633GB.
Every input copy, output, closed bill and harness-only post-context reset
has an immutable receipt. Both-role output/bill seal precedes truth access.
No fit, tuning, new output, checkpoint selection or candidate alteration
was performed during this test. The score-only judge took20.470s.

Result:panel_v2/score/result.json, SHA
767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b.
Checker:panel_v2.py, SHA
e5f16497f4c8c825f9a203dac3e8c0c3ceac218731605c7c818ae5e8a237268e.
Terminal estimator remains
d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451.

Decision: both prospective strict group gains pass. Preserve this as a
local accuracy component. No below4e-9 milestone is reached by the whole
panel; the virgin-group number alone is not a milestone. Production
NumPy-free residual/runtime/RSS and packaging checks remain before any
bank, including a future combined candidate. Gains cannot be multiplied
or added to compression savings without fresh joint validation.

Official bank is unchanged:#329926,4.982231460183304e-9 adjusted.
