# Fixed lazy coefficient rotation on Weight5

Phase 2 only. This is a port of the preserved lazy-rotation component, not a
new raw-error mechanism. The previous goal turn made progress: the actual
non-Gaussian CP gate experiment closed negative. Its gate stays closed.

Parent Weight5 SHA256:
`a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9`.
Reuse the exact numerical fragment from
`astra_phase2_reexamination_v1/current_comp_shared_lazy_rotation_v1/fnp.py`,
SHA256 `e96ee6b660a688f767abd2b8300883e52920eace477e35e7b5edaae759ac6e92`.
The older dense V2 algebra passed; deployment was deferred because the BOX
parent took 51.229 seconds and the component adds calls while saving FLOPs.
Current Weight5 has measured real-client times 39.4935/37.5752/38.2081 seconds.
This material runtime change warrants a fixed current-parent test. No former
BOX runtime claim, raw gain or qualification is inherited.

Keep rank384, age4, the full QR/canonical Gram-core update, signed coefficients,
all 1024 reconstructed rows, every diagonal/omitted-radial consumer, physical
map, learned heads, kernels, arena and Weight5 owner cache unchanged. Defer
rotations only at layers8/10/12/14, consolidate at9/11/13, and use the layer14
map through layer15. At most one map and two factor groups persist per model.
For overlap L and prior factors F0, virtual factors are concat(L F0,Fj).
Reconstruct with concat((U L)F0,U Fj), and assemble the two inner repeated-plane
reads before the original common trailing multiplication. This changes FP32
association, so output parity or raw accuracy must be measured.

Preserve all fragment layout/finite/overflow guards and its paid scans. Build
append-only from Weight5, replacing only the last definitions of shared state,
push and read. Prove each replacement invertible and the parent byte prefix
exact. No fitting, new labels, new arena, schedule/rank/precision sweep or change
to existing pending archives.

First check the fixed fragment against independent FP64 materialization with
rank384, noncommuting maps, nonzero signed pairs, immediate consolidation and
two final-map consumers. Original tolerances: maxabs <=2e-5+5e-5*maxabs(reference)
and relative Frobenius <=2e-4, denominator floor2^-20. Include invalid layout,
history, stale-factor, nonfinite and envelope rejection cases. These are dense
algebra tests, not deployed packed-kernel or billing certificates.

Then one native35/100/35 and independent cold100 on the actual complete current
candidate. Require finite nonnegative all16 outputs, byte-exact own repeat and
warm/cold100, byte-exact first8 parent layers, all-layer parent maxabs <=1e-5,
and retired Weight5 caches. Seal sources/weights/outputs/complete own bills before
reading the existing ordinary1B development truths. Advance only if both
own-cold adjusted scores improve and pooled adjusted ratio <=.99. Retain the
complete cost J = raw MSE * max(.1,F/2^41), including initialization.

Only a development pass earns the existing 14-network public8/fresh6 panel.
Every prediction is recomputed; previous raw errors are not reused after a
changed association. Require improved pooled noise-corrected adjusted score in
each group and overall, overall ratio <=.99, at least12/14 individual wins,
finite nonnegative outputs and all-layer maxabs <=1e-5 against Weight5. This is
an existing reused panel, not new independent evidence.

If the panel passes, require real client35/100/35, each call <45 seconds,
pooled wall <=1.10 times the recorded Weight5 sequence, repeat parity and native
bills/outputs matching the same call history. Runtime may rise within these
fixed limits because this component targets FLOPs. Follow with contract/domain
and package checks as applicable; do not describe tiny validation as full-width
coverage. Only a fully qualified package becomes a proposed milestone bank.

One numerical process group at a time, nice10, every BLAS pool1, external8GB
RSS and256MiB added-swap guard. Require zero added swap in completed qualifiers.
No new Monte Carlo or large captures/downloads; expected artifacts <10MB before
client logs. Preserve each failure. A failed fixed gate closes this port with
no schedule, rank, window, damping, padding or precision adjustment.

Official best remains #331584 adjusted4.015535762022727e-9. Weight5's exact
archive question remains pending after the earlier Cost75 automatic upload
review rejection. This local experiment is not upload approval. The approximately
1e-9 goal remains active; a small compute gain would not complete it.
