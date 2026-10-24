# Fixed three-packet D5 read kernel

Parent: unchanged Cost75, source SHA
5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c.
The current dispatcher permits read depths 0 and 4. This candidate adds depth
5 with the same shape-only 750,000-F request penalty. No estimator physics,
payload, rank, cohort age, head or fitting changes are permitted.

Execute the original four forward transforms. Partition the first batch axis
of their two leaf operands into lengths 2,2,3. For each packet execute one
Winograd level, then reconstruct directly into that packet of the D4 output
leaf. Finally execute the unchanged four backward levels. This is exactly
the full D5 arithmetic circuit, including operation association, with a
different schedule. Cached bindings contain views/shape metadata only.

Fixed maximum geometry is (1024,8192,1024), aligned to32. The live D4 inputs
occupy floats [0,157351936). Once all four forward levels finish, the old
level3 regions are dead. Child A occupies [157351936,216358912), child B
[216358912,275365888), child products [275365888,282741760). Child scratch
aliases the start of child A only AFTER its matmul, when both child inputs
are dead. Relocate the parent D4 product leaf to [282741760,292576256).
All original four-level backward destinations/scratch are unchanged: the
first backward destination is below9834496; the later destination that
overlaps the relocated leaf is written only after the D4 leaf is dead.
No input or output from another product is kept in the arena. Total arena
grows from292225024 to292576256 floats:1,404,928 extra bytes.

Unit test fixed shapes (1024,8192,1024), (832,8192,768), (32,1056,32),
deterministic independent Gaussian operands and a separate full-D5 arena.
Require finite outputs, byte identity with full D5, repeated identity after
other geometries, relative RMS versus float64 direct product <=2e-4, and
lower warm billed work than D4 on both large geometries. Save all own
cold/warm kernel bills and wall times, plus an explicit interval certificate.
No truth labels are read by this unit test. Correctness bugs may be fixed
before any estimator labels are opened, preserving failed attempts.

On success, run native35/100/35 and independent cold100. Seal outputs and
own cold bills before existing1B development scoring. Require exact repeat,
exact warm/cold100, both adjusted scores improve, pooled ratio <=.99.
If that passes, use the unchanged existing14 panel and real NumPy-free
client. Boundary requires exact all16 output/bill parity, cold and repeat
calls <=45s and pooled three-call time <=1.10 times the Cost75 receipt.
No shape penalty, packet size, rank or head sweep on failure. These are
development/existing-panel tests, not untouched generalization evidence.

Only one numerical group runs at a time under the existing nice10/BLAS1/
8GB RSS/256MiB additional swap guard. No dataset downloads. The Cost75
archive and its pending exact-archive upload approval stay unchanged.

Panel scoring rule recorded before the new panel is run or opened: inherit
the parent qualification rule of improvement in each of public8 and fresh6;
quantify "no large isolated regression" as no individual adjusted-NC ratio
above1.03. Compare directly against frozen Cost75. Any official-parent
projection uses the direct paired ratio to graded Cost150 on those same14
networks, because Cost75 has not been graded. No projection is a bank.
