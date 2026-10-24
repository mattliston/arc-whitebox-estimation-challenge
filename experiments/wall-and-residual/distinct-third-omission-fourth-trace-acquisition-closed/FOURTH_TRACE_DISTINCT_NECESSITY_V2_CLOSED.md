# Distinct-third omission from fourth-trace acquisition — CLOSED

2026-09-06 03:23:51 PDT, Phase 2. Fixed rank35, pre/post layer12. No
targets, new samples, model write, fit, candidate, or score were used.

V1 stopped on an unnecessarily strong PSD precondition for a linear
inverse. V2 separately certified the actual Sylvester/rank-one conditions
at SPD and indefinite synthetic fixtures, maximum inverse error4.52e-17.
The bank covariance was not clipped or altered. The preserved repair note
explains the -1.325734e-6 minimum eigenvalue and algebraic domain.

All12 bank prefix feature/sigma comparisons passed byte-exactly, as did
the actual pair mean, covariance, target plane, and radial output. The
full retained third plane differs from the masked consumer plane by norm
1.93126466; these were deliberately kept distinct in the audit. Full
retained K was used only for removal of its repeated entries, actual bank
K for the pair and lower-support terms. This is not the older full-plane
instrument state.

Omitting exactly the all-distinct input-third contribution, retaining
every covariance edge and full matrix core, gives relative errors:

| Object | Relative error against the full reference |
|---|---:|
| Non-radial fourth trace | 0.8741915784 |
| Covariance-metric fourth core | 0.8357335120 |
| Next scalar fourth mean response | 0.6869008000 |
| Following distinct-third mean response | 0.5903611831 |

Each had to be <=0.10. Full reference norms are4.96955887 for the trace,
0.28836342 for the core,3.884997e-4 and8.415205e-5 for the two response
vectors (norms, not MSEs). Core reconstruction relative error4.94e-15;
source subtraction max3.47e-16. Runtime6.048s; peak hostRSS4,494,131,200B.

The expensive contraction class is material at this fixed current state.
This is NOT evidence that the complete core improves true accuracy, nor
a universal complexity lower bound. No rank100, omission subset/strength/
age sweep, sampling, carrier, fit, native port or submission follows.
The old cost-failed full carrier and failed thin q4/s4 route remain closed.

Receipt fourth_trace_distinct_necessity_r35_v2.json/.npz.
Core helper SHA25627609159d354bfb88f033d5077beffea15b9c833926bc923fbd3145fc6df44e6.
Wrapper SHAb603e2a58f627a2731769def825e39572bdfebe2ab036de4a119ab1397cf2618.
Read-only official bank poll around03:22PDT: #329926 graded, adjusted
4.982231460183304e-9, raw2.2448352261505987e-8. No completion-count audit.
