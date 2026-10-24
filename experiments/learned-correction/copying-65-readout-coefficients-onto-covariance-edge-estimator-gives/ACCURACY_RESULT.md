# Fixed covariance/readout composition: native result

The fixed combination passed its prespecified eight-network accuracy gate.
Pooled adjusted/covariance-bank ratio is **0.995858110143**,
with5/8 individual wins and worst ratio1.004219075102.
Parent pooled adjusted is3.6242291716621936e-09; composition is3.609218013617435e-09.
Every complete cold operation count is exactly the bank's.

| Network | Raw/adjusted ratio | Complete cold FLOPs |
| --- | ---: | ---: |
| 37 | 0.998019431551 | 416772131898 |
| 39 | 0.981771106430 | 431313688249 |
| 42 | 0.990562483215 | 414443321465 |
| 48 | 0.994769747959 | 415976509965 |
| 59 | 1.003515583802 | 413799356015 |
| 63 | 0.989532289242 | 403809864481 |
| 66 | 1.004219075102 | 393619641825 |
| 67 | 1.001106022010 | 410324733951 |

Only the already served331682 final64 head weights and bias were copied into
exact331684. All other packed parameter bytes match; reversing the two field
edits restores the whole bank source. No fitting, tuning, interpolation or new
reference data was used. The current-edge nonlinear528 payload was not used.

Bank35/100/35 reproduces all16 saved rows and complete bills. Composition
35/100/35+cold100 preserves the first15 rows, final hidden state, scalar host
and sigma exactly. All own repeats and warm/cold100 outputs are byte exact.
Three actual-FNP readout fixtures match independent FP32/FP64 arithmetic.
All15 native calls have finite nonnegative FP32 outputs and positive unfloored
variances, exact host+delta reconstruction and retired caches. Eight cold
confirmation outputs and bills were sealed before target decoding; each
first15 rows and bill equals its authenticated bank baseline. NumPy MSE is
independently checked by math.fsum, max relative difference
1.861e-16.

The eight networks are outside both incremental fits, but are reused research
networks, not pristine holdouts. They do not establish an official score.
This passes to separately sealed ordinary-client, runtime, contract and package checks under the standing upload authorization.

All16 guarded groups retired. Fifteen full calls used
6142986229433 FLOPs; peak group RSS
6437601280 B, self-reported
6437912576 B; zero added swap. No GPU, downloads,
new samples or fits. The official bank remains331684 at adjusted
3.5844144212297072e-9, raw1.949715441540434e-8,100/100 complete,zero failures.
4e-9 is banked;3/2/~1e-9 remain unmet and the full goal is active.
