# CLOSED — learned endpoint-history covariance decoder

September6,2026, Phase2. Eight coherent8M teacher trajectories replayed
with every archived basis/rho/sigma/target/weight/response array byte-exact.
Only frozen P0 recurrent histories were added, always before current truth
feedback. No new MC, private labels, state writes or submission.

Fixed65 endpoint features (64 learned states plus constant), symmetric
Theta0/1/2 shared over late layers8..14, multiplied by R^0/R/R^2. Six
train35/100/150/200/250/300, held-out400/500. Same100-epoch objective,
initialization/seeds/LR/ridge as designed. Final epoch only,7.83s MPS.

| Held-out net | Covariance MSE / zero decoder | Next-mean-response MSE / zero decoder |
|---|---:|---:|
| 400 | .7414121563 | .9933264149 |
| 500 | .7444433204 | .9855317053 |

Information gate required BOTH ratios<=.75 on EACH net. FAIL. Learned
histories captured more covariance Frobenius error than the old local
kernel, but almost none of the error relevant to the next mean. This is
not a claim that every learned pair architecture is impossible, nor that
the present fit is an exact representation ceiling. The fixed decoder is
closed: no extra epochs/ranks/whitening/features, covariance write, or
expanded fit. Archive receipt learned_pair_history_verdict_v1.json.

Decoder wrapper SHAe3dd7c2630cfb4555facbf124136a2327242364b23c4b21d60b87e4ac14c1479.
Frozen P0 SHA166b3c67ec4d27793d433e38e15086b95d052426fdf435a79067cc07a2a4dd7e.
