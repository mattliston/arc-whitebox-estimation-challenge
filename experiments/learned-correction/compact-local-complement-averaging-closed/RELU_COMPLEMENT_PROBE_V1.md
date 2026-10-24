# Exact ReLU complement identity: bounded correction probe

September5,2026. For every scalar Z, relu(Z)-relu(-Z)=Z. The current
Gaussian/first-order cumulant mean obeys the same identity. On the honest
teacher-forced prefix, its pre-mean equals the stored truth prefix pushed
through W, so the fresh mean residual is even under global Z->-Z at a
single layer. This supplies an exact same-target partner without sampling.

Freeze the current bank's full-pair moment host and learned head. At each
layer independently, BEFORE current correction, evaluate an auxiliary
negative branch using -mu, the same Cpre, -K21, the same radial fourth
planes, and -W. Use the original positive branch's incoming recurrent
state for both heads and carry only the positive state. This is a local
diagnostic feature construction, not a second altered network trajectory.
Important: reuse the actual supplied K21 plane, including its compact cold
mask. Do not claim this is the negative branch of a rerun with reselected
active neurons. Mean complement remains exact despite this approximation.

Ranks35..58, layers10..15, held-out37/39/42/48. Verify parent host/features,
sigma and positive delta byte-exact; verify m_plus-m_minus=mu to1e-12.
Compute frozen-head delta_sym=(delta_plus+delta_minus)/2. No fitting,
no coefficient sweep. Require >=5% held-out mean residual MSE reduction
versus parent AND improvement on >=3/4 held-out nets. Otherwise close,
without200 rows or recursive port. A passing information gate still needs
matched closed-loop and final-head handling: on-policy final residual is
NOT even when the incoming mean is imperfect. Do not symmetrize the final
on-policy head blindly. Actual cost/runtime and14panel remain required.
