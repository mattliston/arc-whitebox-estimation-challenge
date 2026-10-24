# Bounded learned joint-graph information test

September5,2026, after wall-repair bank329926 acceptance. Phase2 only.
No board-score claim and no 200-net training authorized by this pilot.

Prior fixed20 correlation summaries and fixed48 response-plane features
failed their gates. They did not learn nonlinear messages from the full
126-dimensional recurrent-head input through the actual joint matrices.
This pilot tests that specific extra information, not a capacity sweep.

Freeze the current full-pair host and mean head. Generate24 teacher-forced
trajectories, ranks35..58, with exactly parent host/sigma/original30 features.
Capture head input h and standardized pre-covariance R and third plane T21
at layers10..15 BEFORE current truth feedback/mean correction. Remove
diagonals. Labels are frozen-parent mean residuals against stored1e9 means.
No current-target-dependent state or features; parity and late profile gates
must pass before training. Preserve every existing row and receipt.

Fixed20 train nets and4 held-out37/39/42/48. Fit one40-epoch residual graph
head and a same-capacity local control, same seed/optimizer/schedule. Both
encode h126->32->16. The graph arm aggregates16 learned channels over
R, R squared, T21, and T21 transpose (row-L1 normalized); the control uses
identity operators. Concatenate h,32 local features,64 messages;64-hidden
SiLU output with zero-initialized last layer. Scale by sigma and parent
layer RMS. No hyperparameter sweep or selection by best validation epoch.

At the fixed final epoch require graph held-out MSE <=.92*local-control
and <=.95*frozen-parent, plus improvement over parent on >=3/4 held-out
nets. Otherwise close this block without200 rows, recursive port or fire.
Success is only an information gate; it still requires200-net fixed-split
fit, matched closed-loop35/100 checks, scalar-final refit, full14panel,
runtime/residual/memory and packaging. Pilot scores are not bank scores.
