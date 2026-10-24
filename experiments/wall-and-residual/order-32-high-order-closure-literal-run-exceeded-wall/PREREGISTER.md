# A1_NO_CP4 wall-safe pure-closure successor

Status: **target-free source qualification in progress; no target, score,
dataset, sampling, fit, submission, graph mutation, or scientific promotion is
authorized by this file**.

## Question

Can the frozen Phase-2 V4 closure remove the complete CP4 carrier/history while
preserving every other scientific component and making the official runtime
robustly wall-safe?

## Parent evidence and trigger

The immutable parent is
`../fullbudget_k21_k4_closure_v4/estimator.py`, 68,698 bytes, SHA-256
`3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c`.
Its common-driver meter used `2,179,742,954,469` FLOPs, 35.341105 seconds wall,
0.392632 seconds residual, and 4,618,567,680 bytes descendant RSS.  Board
submission 328199 smoke-passed and completed MLPs 0 and 12 with final raw MSEs
`7.661e-8` and `7.071e-8`, but production wall time of about 117--124.6 seconds
caused `TIME_EXHAUSTED`.  Those board observations trigger this engineering
successor; they are not opened or used by its source or tests.

The target-free accuracy-frontier design is frozen by
`../fullbudget_k21_k4_closure_accuracy_frontier_v1/PREREGISTER.md`, SHA-256
`e936c2ff8d526605173cf268d27025c660847d4649508bbe9a1fdc163139654e`,
and `VARIANT_MATRIX.json`, SHA-256
`c2db61a2c720cadfc53a0e980626cc6f654c94e224037f98cf27eb357dbf33da`.
It specifies A1 before any local target score.

## Frozen scientific change

The candidate removes CP4 as a complete state family:

- no CP4 rank/cap constant, lambda or factor state;
- no push, preconsumer plane read/add, nonlinear gate, coordinate birth,
  top-energy sort/truncation, dropped-plane read/complement, post-provider
  subtraction, stored-float32 CP4 rounding, or W16 diagonal read/add; and
- no empty or zero-valued CP4 side path retained for accounting or dispatch.

The coherent `Star4(H4) + alpha P_C` residual directly fits the complete
corrected `(K4,K31,K22)` reset at h1 and every later reset.  Its affine push is
the sole quartic provider before Q32 and at W16.  CP3, H3, H4+alpha, L4,
`1/2 L3^2`, Q38 coefficient construction, Q32 pair series, spectral
diagonal-preserving PSD repair, W16 marginal specialization, dtypes, guards,
and layer order remain unchanged.

The preregistered mutable candidate draft at the time of this file was written
is `estimator.py`, 63,205 bytes, SHA-256
`5b222ef4f8120adb9443da5c2a1f89c79085a768ac2aef10c61c18413ef6a025`.
Any later source correction must be disclosed before qualification and receives
a new immutable HOLD hash.

## Legal inputs and forbidden access

Qualification may use only synthetic fixed matrices, target-free source
inspection, the pinned local FlopScope/Whestbench environment, and existing
immutable source/resource receipts.  Public dataset members and targets,
truth/teacher arrays, local score files, fits, samples, board actions, and
canonical graph writes are forbidden.  The production meter may run only after
the root coordinates the shared heavy-compute lane.

## Qualification gates

1. The source parses and imports exactly `flopscope`, `flopscope.numpy`,
   `whestbench.BaseEstimator`, and `whestbench.domain.MLP`; array-valued
   numerical work remains inside FlopScope and no forbidden mechanism appears.
2. A deterministic parent-to-child diff shows only complete CP4 deletion,
   direct H4 ownership of the corrected quartic reset, the necessary CP3 gate
   cast relocation, tuple/control-flow repair, and truthful documentation.
3. No CP4 state or computational seam survives in the AST.  CP3 rank/cap,
   push/read/grow, bank aliasing, NaN-poison, and off-shape guards retain their
   parent semantics.
4. On four fixed depth-16 synthetic fixtures, A1 is bitwise equal to an
   independently spelled intended-science reference that disables the full CP4
   family and routes the complete quartic reset directly through H4+alpha.
5. Tiny algebra fixtures prove direct H4 reset reconstruction: diagonal K4 and
   directed K31 are replayed up to declared fp32 H4 storage rounding, while
   alpha is the same one-dimensional K22 pairing projection.  Provider add and
   subtraction counts are zero.
6. The projected production deletion binds the frozen CP4 audit's enumerated
   `234,507,794,432` FLOPs (dominant kernels `233,997,074,432`) and at least a
   10.7% reduction relative to the V4 literal bill.  Exact calls, whole-source
   FLOPs, RSS, wall, and residual are claims only after an identical-driver
   literal meter.
7. The literal production meter must return finite float32 `[16,1024]`, use at
   most `2^41` FLOPs, wall below 120 seconds, residual at most 0.4 seconds, and
   descendant RSS below 8 GiB.  A wall-safe promotion additionally requires at
   least 10.7% fewer FLOPs than V4 and materially wider residual/wall margin.
8. The author freezes only after all non-meter gates pass; an independent
   auditor rehashes immutable HOLD bytes and may reject or request a new
   additive version.  No accuracy or board promotion follows from source or
   resource qualification alone.

## Outcome split

- All source/algebra gates and the literal meter pass: freeze A1 as a
  Phase-2-legal target-free wall-safe candidate for the already authorized
  fixed-roster accuracy evaluation.
- Science-reference parity or reset ownership fails: reject these bytes and
  repair only in a new disclosed source version.
- Resources fail despite complete CP4 deletion: close this spelling as an
  engineering failure and activate the next preregistered wall-safe successor;
  do not reintroduce partial CP4 planes.

## Negative claims

This experiment does not establish raw MSE, local/board ranking, moment
feasibility, target generalization, or that CP4 is universally harmful.  It
tests one exact coherent A1 host and preserves the A0/A1 causal contrast.
