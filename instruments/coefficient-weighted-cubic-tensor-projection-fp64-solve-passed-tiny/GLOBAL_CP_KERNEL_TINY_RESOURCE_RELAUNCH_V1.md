# Resource-only relaunch of the unchanged tiny v2 test

September 7, 2026, 01:49 PDT. The restricted launcher printed
"nice: setpriority: Operation not permitted". V2's nice assertion stopped
before fixture generation: tests=[], nice=0, wall0.00458s, native FNP0.12.0.
That failure is preserved as global_cp_kernel_tiny_v2.json; no numerical
data was produced or inspected and no computational retry law was used.

Relaunch the exact unchanged v2 source with an approved nice10 shell. Only
its output path is assigned to global_cp_kernel_tiny_v2_nice10.json by the
launcher. The source still seals itself and the three prospective notes.
No source, fixtures, thresholds, precision, solve, seed or reference changes.
No numerical relaxation or new network/target access. No overwrite.
