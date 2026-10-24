# End-to-end recurrent encoder direction: bounded information test

September 5, 2026, 21:17 PDT. Phase 2 only, no new inference operations.

The closed910-parameter pilot adjusted ONLY the per-layer mean heads.
This test leaves those heads, gains, normalizers and the final scalar head
frozen. Its distinct state action changes only the existing recurrent
message transform and three-layer feature encoder: msg_w/msg_b and
enc0/2/4 weights/biases. It does not widen the model or add features. The
earlier head-only radius/epoch/refit family remains closed.

Use the certified CPU full-carrier derivative, float64 moments and
float32 carrier/learned arithmetic, with actual dynamic selections.
Training networks35,36,38,40; fixed holdouts37,39,42,48. They are held out
from this new direction, not claimed unseen by every historical fit.
Labels are the existing1B means. One CPU worker, three torch threads,
nice10 launch explicitly permitted; no MPS or concurrent card jobs.

Compute the average gradient of uniform final MSE/2e-8 on the four train
nets. Precondition each existing parameter group by its original mean
square, floored at1e-8. Normalize the resulting negative-gradient
direction so the largest group-relative Frobenius norm is one. No
adaptive family of feature groups, optimizers or gradient directions.

Obtain the exact local-branch output tangent along that one direction on
all eight nets. Choose the least-squares scalar step using TRAIN outputs
only, clipped to[0,.10] so no group's relative update exceeds10%. Check
reverse/forward derivative agreement. Export only that single payload.

Judge the finite update in the independent NumPy actual bank reference
on the four holdouts. Expansion requires mean raw ratio<=.90 and at
least3/4 wins, with finite outputs and torch/NumPy discrepancy<2e-6.
No other scalar, intermediate checkpoint, optimizer, radius or group
selection is chosen after seeing holdout data. Failure closes this
particular encoder-direction probe; it is not a universal NTK ceiling.
No submission follows without full independent panels and release gates.
