# Fixed BFIX gamma0.125: rejected after TRAIN8

Candidate d13c9262352f6db0927aaacd0c0ffb78dba8ccdaa2bbe0babce2cd22b9ded83d adds only `_BFIX_GAMMA = 0.125` to official S33 source88683. It revisits a neighboring scalar with a documented old positive mean and worse tail, preserving the original history.

TRAIN2 passed: adjusted−0.6025446646%, raw−0.6025747663%,2/2 wins, identical bills. Early boundary passed all fixed gates: projected residual0.343850751s, setup3.32s, full native bills exact.

TRAIN8 failed: adjusted−0.0710003946%, raw−0.0705015685%, identical bills,5/8 wins. Paired95% CI[−0.4722029478%,+0.2972412935%] crosses zero. Additional6 adjusted+0.0842479843%,3/6 wins. Thus wins, negative CI and independent extension gain gates fail. Integrity passed: all16 finite, L0 byteexact, source/setup/complete bill/UBC ownership all checked. No retention, broad validation, serving qualification or submission.

All guards completed and retired, zero added swap. Native2:50.2151s,5.224GB. Boundary:65.1030s,5.739GB. Native6:144.9967s,6.591GB. Sessions55433/6548/8430 reaped. Assessment guards also passed.

Conditional gamma0.15 source was assembled before the N8 outcome, with a hard parent-pass prerequisite. That prerequisite failed: preserve the source unexecuted; no follow-up gamma, scope or fit rescue. The retention recorder draft is unexecuted and has an identified sorted-rank assertion issue; it is not approved for reuse.
