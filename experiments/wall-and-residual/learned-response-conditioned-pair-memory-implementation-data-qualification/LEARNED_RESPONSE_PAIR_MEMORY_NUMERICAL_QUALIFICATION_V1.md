# Learned response-conditioned memory — implementation/data prerequisites

September6,2026, Phase2 only. These are NUMERICAL and provenance results,
not learned information, raw score, rollout, metered cost or a new bank.
The pure10-parameter model and accepted protocol remain frozen. History
and RESET are both qualified; RESET has8 structurally active parameters.

Synthetic n7/history13 certificate finished23:44:56UTC, wall1.68653s,
peakRSS259,751,936B. CPU64/NumPy64 outputs agree to3.13e-17 relative;
gate-gradient finite-difference disagreement <=1.18413e-10. MPS32 output
disagreement <=1.92304e-8, gate-gradient disagreement <=1.21390e-7.
Response-unit identity error4.82594e-16; query-sign parity and bounds pass.
Only the single fixed fixture/vector/probe was used, no sweep or retry.
Receipt SHA df4dfa5a0cd4e9410c07aa34e6426538c2acfd585e1eddbaafb42ac0b0939b56.

Rank35 causal preparation passed with no bank rerun, wall.371531s,
peak512,606,208B. Its global completion SHA is
551ca4f6f4f80da11685bb2689433069ac88f5b03901a73d55fa7268d6615c81;
input NPZ SHA7d095fe59f98a458974798fb1efbc09e45799ea2bd7ec02ba7f04ab4119a76b4.

The inputs-only rank35 full-width certificate finished23:50:21UTC,
wall3.45221s, peakRSS2,590,736,384B. Both NumPy64/CPU64 outputs agreed to
4.21328e-16 relative. MPS32 output <=2.22371e-7, full gradient <=2.36845e-7,
gate-gradient <=4.90093e-7. No loss.npz was opened or hashed by this
certificate; its probe was synthetic, not a target. Receipt SHA
e1ea3d1f919504ec886436a48b6a50a6db34b5f443fdb09eb1cccd48b6f81663.
The comparison is on preparedFP32 causal inputs, not an independently
regenerated native-FP32 moment carrier or a deployment parity claim.

All remaining23 fixed rows36..58 passed preparation, each about.32--.42s;
maximum processRSS1,526,759,424B. Every row seals exactly eight causal
fields before deriving its separate teacher-action loss file. The full
old capture/source/P/Q/weight chain and final preparation hashes passed.
Remaining completion SHA
854f1165668af9531d1eb33c7d1683d6dba0a18a99d1598c113c068109c632e6.
Preparation source SHA
292949c430daa1a7f2b9b07ccdafc063d31fc8c2dcf11d6fc8b65cac1555e558.

Numerical programs ran nice10, requested twoCPU threads, and used only
one MPS worker at a time with fallback0. AP never imported threadpoolctl.
Last memory-pressure check was816.75MiB swap in use. No thresholds were
altered after observing data. No fit has been scored by this note; the
paired100-epoch fit and its .50/.50/.80/3-of-4 gates are the next stage.
