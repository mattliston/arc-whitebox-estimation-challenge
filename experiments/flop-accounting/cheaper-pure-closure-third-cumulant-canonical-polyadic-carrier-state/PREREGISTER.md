# Q24-native CP3 age-retirement discriminator

This bounded observer-fit rank-0 discriminator asks whether the full-birth
third-cumulant carrier can retire old 1,024-column birth cohorts without
destroying the near-`3e-8` native-J result.

The frozen arms are the full D1 carrier and chronological windows
`w = 1, 2, 3, 4, 6`.  At a saturated transition, the complete currently
carried CP state is still read into the current preactivation K21 plane.  Only
after that full Q24 reset is the oldest cohort retired.  The retained suffix is
used in the post-ReLU K21 reconstruction together with the exact new
coordinate birth, so `_reset_stars` transfers the retired discrepancy into the
dense cubic-star residual.  The retired cohort is not silently deleted before
the reset.  This is the cubic analogue of the incumbent `_cp4_reset`, which
subtracts dropped CP4 planes from its retained provider and lets
`_reset_stars` absorb the difference into the quartic residual.  The first
retirement in every arm also compares split `retire + keep` K21 against an
unsplit read at the same current reset.

Each arm runs in a fresh process.  Direct Q24 output, the unchanged frozen
native-J correction, physical local q errors at zero-based layers 10 through
15, literal diagnostic FLOPs, and peak RSS are reported.  The sole development
target is the already-open observer-fit rank-0 target embedded in
`public_phase1_kerdock_hidden_snap_v1/runs/row_0000/DIAGNOSTIC_ARRAYS_BASE.npz`.
No observer-selection data, public identities, board scores, network access,
training, fitting, or submission is permitted.

The frozen native-J `raw20` rank coordinate retains its original logical
full-birth value `min(1024 * layer, 15360) / 16384`, and its retained flag
stays one: retired mass is represented in the cubic star, and substituting the
smaller physical CP factor count would create a large artificial OOD feature
shift in the frozen checkpoint.  The physical carried ranks are recorded
separately in the receipt.

Fail fast for the compute route if neither `w <= 2` keeps frozen-native-J
final MSE within 10% of full D1 nor `w = 3` keeps it within 5%.  A passing arm
licenses a disjoint observer-fit panel and an exact production meter; rank-0
alone is not a deployment claim.
