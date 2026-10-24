# Positive-path matched pilot closes without an expansion

September 5, 2026, Phase 2. All40-epoch final checkpoints,20 train/4
validation, matched prefix and final-head refits, actual NumPy replay.

Pooled final MSE against existing1e9 truths on37/39/42/48:

- Original bank reference: 2.12016561136e-8.
- Refitted parent control: 2.23546349938e-8.
- Refitted positive-path source: 2.18539989183e-8.
- New/control .97760481996; new/bank1.03076848343;1/4 bank wins.

The fixed .90/.92/3-wins gate fails. Both branches' final inputs, hosts
and sigmas match their frozen on-policy training rows bytewise. No200-net
expansion, alternate epoch/strength/age/source selection, panel or upload.

The large internal source result remains real but insufficient. A fixed
independent-two-stream rank35 post12 consumer diagnostic gives:

- Before mean feedback, covariance signal-error ratio .6777317441.
- Next Gaussian-mean linear-response error ratio .9697370369.
- With the fixed1e9-mean consistent covariance update, the corresponding
  ratios are .9799114974 and .9454709781 (diagnostic oracle update only).

Thus covariance Frobenius gains mostly miss the next value-bearing
directions in this cell. This is not a global impossibility result.
Preserve the exact24-term and positive-factor certificates; no new score.

The isolated FNP.12 birth-factor implementation is byte-identical to the
NumPy algebra on its wide test. Its projected incremental15-layer bill
is849,991,680 FLOPs, .17324% of the rank35 bank bill. Cost was not the
reason for closure. The current official bank remains329926 at4.98223e-9.
