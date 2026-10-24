# Submission 329851: terminal graded result is unusable

API result observed September 4, 2026 at approximately 23:07 PDT:
status `graded`, message `Graded successfully`.
Adjusted score: **0.6889726864485142**.
Raw score: **0.6889726907301852**.

This is not a successful bank. Matt's earlier live-card observation was
23 scored, 5 in flight, 49 failed. The aggregate is consistent with failed
MLPs dominating the result, but the API provides no per-MLP failure causes
or failure counts. Do not infer a clean run from the word `graded`, or
report this as terminal API status `failed`.

The archived local independent 14-net adjusted score 4.657375968018e-9
describes only successful local accuracy/cost measurements. It is not
the official result. Earlier acceptance and local release receipts remain
preserved as narrower historical evidence, superseded by this result.

Official best remains 329790, adjusted 1.436993412541317e-8.
No promotion, commit, or duplicate upload. The below-4e-9 goal is unmet.

Diagnostics: original local server NumPy 2.4.6 uses Apple Accelerate.
An isolated PyPI macOS-11 ARM64 OpenBLAS wheel with the same NumPy and
FlopScope versions was created, without modifying existing environments.
Unchanged candidate, rank 35, one thread: 43.042 s wall, 0.194850 s residual,
6.861 GB max child RSS, versus 34.518 s / 0.191471 s with Accelerate.
This exposes a portability gap but does not by itself identify the
official failure. A five-MLP persistent remote-process test is underway.
