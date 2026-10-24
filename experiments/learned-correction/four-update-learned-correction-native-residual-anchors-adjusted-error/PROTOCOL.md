# S34 native-anchored small4

This is one fixed empirical learning test from official S34 (#333050), source
`42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb`.
It does not reopen the failed warm2 endpoint or failed new-message anchored fit.

Use sixteen distinct TRAIN networks in this fixed order:
38, 40, 41, 43, 44, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 60.
There is one pass, four Adam updates, batch four, and one final endpoint.
No validation, N2, N6, or final targets enter training. There is no endpoint,
coefficient, epoch, roster, or learning-rate selection after observing outcomes.

Extract all initial arrays and physical dials from pinned current S34. Train
the existing 3,437 main/head/edge scalars and ten physical dials. Freeze the
44,144 trunk scalars, five normalizer arrays, head rows 0 and 15, coefficient
columns 1 and 3, and X3/c1. Preserve the original main/dial learning rates,
three-update warmup, 120-step cosine schedule, separate norm-one clipping,
and per-network loss scaling 1e8/4. The 120 steps describe the cosine schedule,
not the warmup length. The proven import-time gradient/thread repair is
integrated directly into this trainer.

The root obtains sixteen genuine native S34 predictions under the original
7.5 GB / zero-added-swap guard. Before optimizer update zero, compute and save
all sixteen Torch initial predictions and fixed targets:
`target = truth - S34_native_initial + Torch_initial`.
This aligns the initial loss residual with current native S34. It does not
certify the Torch/native Jacobian: documented projection, cropping, precision,
dead-side, YNT, and other structural gaps remain. Anchoring previously failed
with new message parameters; the present frozen-trunk existing-parameter test
has no presumed gain.

Training is serial, one thread, at most 900 seconds and 9.5 GB RSS, with zero
added swap. It saves one endpoint only. Export changes only the five learned
and six physical-dial assignments, checks frozen bytes, and verifies exact
source recovery after undoing these assignments. Native serving retains the
7.5 GB limit. Existing N2 gates remain at least 0.1% adjusted gain, at least
one win, maximum 1% individual regression, all finite, and L0 exact; continuation
requires the existing fixed-eight gates. There is no broad or serving authority.

`prepare_protocol.py` is stdlib-only and binds the completed source-pinned native
controls into inactive `protocol_draft.json`. Root reviews and activates a
separate `protocol.json`; numerical execution remains solely with root.
