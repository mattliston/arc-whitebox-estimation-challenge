# Fixed current-S32 birth-direction FP32 component

This single candidate ports E-29's previously optional birth-direction internal
precision boundary to the current three-leg S32 implementation. Its parent is
`../hill_climb_star_projection32_v1/candidate.py`, SHA256
`f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef`.
No exact-bundle or other new component is included. The existing third-order
FP32 reductions, learned literals, covariance moments, selection controls and
physical FP64 delta formula remain unchanged.

E-29's older native four-network evidence was roughly neutral incrementally:
approximately 0.10% lower bill and 0.095% higher raw error versus third alone.
It was marked optional, not banked or rejected by a recorded current-parent
gate. This is a new measurement on S32, not a claimed historical gain.

The fixed precision boundary and static storage inventory are in PRECISION.md.
`build.py` imports only standard-library source tools, changes exactly two
active function definitions, verifies exact inverse restoration of S32, and
checks that the complete physical-delta tail remains byte-identical.
`build.json` pins parent, candidate, builder, protocol, inventory and references.
No model or numerical job is run by the builder.

Root alone executes one fixed screen, serially with RSS strictly below
7,500,000,000 bytes, zero added swap, and completed/reaped guards:

- TRAIN2 ranks 35,100: advance only if paired mean adjusted change versus S32
  is at most -0.10%, with 2/2 wins. Both L0 output vectors must be byte-exact.
- On advance only: fixed six ranks 36,45,52,58,64,70, then fixed extra eight
  ranks 3,12,40,41,43,44,57,68. No additional extension or source choice.
- Retain on combined TRAIN16 only if mean adjusted change is at most -0.10%,
  at least 10/16 wins, paired 95% CI upper bound below zero, and positive gain
  on the additional fourteen networks. L0 remains byte-exact on every network.

Existing S32 controls are `hill_climb_star_projection32_train2_v1`,
`hill_climb_star_projection32_train6_v1`, and
`hill_climb_star_projection32_extra8_candidate_v1`. Inherited time limits:
180 seconds / 330 seconds / 400 seconds for those three blocks. No alpha,
layer, precision-island or coefficient sweep. Full bills, downstream state
changes and runtime must be measured; static precision savings are not a
whole-network certificate. Component retention is not release qualification.
