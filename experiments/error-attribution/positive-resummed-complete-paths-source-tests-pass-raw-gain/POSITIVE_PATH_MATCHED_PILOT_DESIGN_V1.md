# One bounded matched co-adaptation pilot

September 5, 2026, Phase 2. Conditional on both rank35 and100 passing the
predeclared independent recursive source gate. The old readout worsens
final MSE, so it is not a candidate. The independently measured source
improvement, rather than a coefficient search, justifies this new test.

Use ranks35..58: hold out37,39,42,48; remaining20 train. Two branches:
unchanged parent control and positive-resummed path source. Reuse archived
honest control rows; generate exactly24 new honest source rows. Existing
feature schema and pre-correction capture order stay fixed. Inspect the
new L10..15 one-step profile for leakage before training.

Stage1: same inherited GNN architecture and bank initialization, same
training split/order, 40 epochs, AdamW5e-4 to5e-5, weight decay2e-5,
existing normalized layer loss, final epoch only. No new features, target
variance heads or end-to-end approximate derivatives. Export each final
mean-only payload. One MPS job at a time; three nice10 CPU row workers.

Stage2: for each branch freeze24 on-policy final-input rows with its own
new prefix payload, no truth feedback. Fit the same scalar final head
from the original bank Q, 40 epochs, AdamW5e-4 to5e-5, same20/4 split,
final epoch only. This is one two-stage pilot, not repeated DAgger.

Judge actual scalar-final closed-loop MSE on the four holdouts using
their existing 1e9 truths. To expand to the full200-net matched round,
require new/control pooled final MSE <= .90, new/original-bank <= .92,
and improvement over the original bank on at least3 of4 holdouts.
The 14-net public/virgin panel, complete FNP cost/parity, persistent
wall/residual/memory tests and official grading remain later gates.

No tuning of source, strength, birth rank, layers, epochs, split or
validation checkpoint on failure. Preserve both branches and receipts.
