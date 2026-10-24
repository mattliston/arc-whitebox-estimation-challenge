# Correction receipt — Qcore reuse history

This corrects the novelty/history claim in `QCORE_REUSE_STATIC_PRICE_v1.md`; that file is retained unchanged as an audit record.

**The mechanism was already tested.** Root identified the exact prior record at TEAM_LOG line5179 and `astra_oct18_bill_v1/BILL_LANE_CLOSES_v1.md` line12: Qcore-alone first-two comparison had bill -0.095%, raw score +0.054%, net adjusted score -0.041%. It was HELD for a later stack and did not receive a standalone32 comparison.

The builder `astra_oct18_bill_v1/build_qcore_reuse_v1.py` is an earlier04Z artifact, not work by another currently active agent. My description of it as contemporaneous was incorrect. The narrow prose search missed the existing named result and I inferred timing without checking its recorded history.

The static arithmetic price remains valid: removing two dense384³ products saves226,197,504 arithmetic units and two matmul calls. **The novelty claim is withdrawn.** The empirical adjusted improvement was much smaller than the arithmetic percentage because the reassociation increased the raw score.

Future exact-mechanism checks must search TEAM_LOG case-insensitively for the exact mechanism name before classifying an existing artifact as new or contemporaneous.

No new candidate or numerical job was created in this correction. Work stopped at root's instruction.
