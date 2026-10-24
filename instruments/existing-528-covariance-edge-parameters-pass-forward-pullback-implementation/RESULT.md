# Existing covariance-edge parameters: implementation qualified

The exact official331760 source initializes all parameters. Only the existing
five covariance-edge leaves (528 FP32 values) are eligible for learning.
All P arrays, A, cstar and edge normalizers/scales stay fixed. The saved source
inverse recovers the prior model_v2 exactly; only trainability and checks change.

The7-node independent forward check has relative error
1.15220872002e-07. The largest independent analytic pullback
error is1.85174227315e-07, below the
fixed2e-5 gate. Four full calls (frozen35, gradient35/100/35) preserve frozen /
checkpointed outputs and repeated35 prediction, gradient, variance, selector
and state bytes. All parameters remain unchanged during these calls.

The real528-leaf Adam boundary, settings and RNG round trip exactly. A second
identical synthetic-gradient update agrees byte-for-byte. These two synthetic
updates are a transport check and are not an eligible model or fit.

Both guards retired, peak5.371462GB, zero added swap,
and post-graph retirement below2GB. No reference mean array was opened, no
production candidate was exported, and no accuracy or native-derivative
qualification is claimed. The separate fixed training stage must initialize
from the official parent, never the synthetic qualification checkpoint.

This is related to the older failed16-output native SPSA and joint covariance
experiments; their negative results remain preserved. Training and actual
native validation are still required. Official331760 remains3.507738856975084e-9.
