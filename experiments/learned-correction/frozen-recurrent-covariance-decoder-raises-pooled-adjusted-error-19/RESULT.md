# Frozen history covariance decoder fails the native development gate

Replacing the bank's covariance edge at layers 8 through 14 with the stored
history decoder increases pooled adjusted error by **19.7306%** on networks
36 and 38. Both networks regress by more than the permitted 5%. This fixed
endpoint stops at its predeclared gate; it does not advance to a panel,
confirmation, package, or upload. No scale, layer, precision, or model tuning
was performed after the result.

| Network | Parent raw MSE | Candidate raw MSE | Parent cold FLOPs | Candidate cold FLOPs | Adjusted ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 36 | 1.983987470157e-08 | 2.250638762958e-08 | 420159401355 | 425163919053 | 1.147913558190 |
| 38 | 1.522485110651e-08 | 1.899279269174e-08 | 414511808707 | 419516326405 | 1.262547520987 |

The pooled adjusted ratio is **1.197306034380315**. Candidate cold FLOPs increase
by 5,004,517,698 on each network, about 1.2%. The score uses each arm's own
complete cold bill and max(0.1, FLOPs / 2^41). Independent math.fsum checks
agree with the raw MSE calculations within 1.75e-16 relative error. These
networks are outside the decoder's six original training and two validation
networks, but were reused in earlier research. They are not pristine holdouts
and this is not an official score.

The endpoint computes a separate recurrent state with the original frozen
P0 encoder on its current physical inputs. Its three saved symmetric 65 by
65 matrices form a covariance correction modulated by 1, R, and R squared.
The old edge is replaced at its existing location. The parent mean encoder,
other parameters, physical backbone, and readout remain unchanged. The
source inverse reconstructs the exact bank and all exported tensor bytes
match. This result concerns this frozen autonomous transfer, not all history
models or covariance methods. The input-state shift from the original
oracle-state training law is present; this experiment does not isolate it
as the cause of the regression.

Formula and implementation qualification passed before final-mean scoring:

- Three deterministic kernel fixtures passed NumPy FP32/FP64, permutation,
  symmetry, repeat, zero-coefficient, and charged parameter-load checks.
- Disabled calls on 35, 100, 35 match all 16 original output rows and each
  corresponding complete cold or warm bill exactly.
- Enabled calls preserve the first eight parent output rows exactly. The
  repeated 36 and separate cold 38 reproduce output and capture bytes.
- All 60 saved recurrent states match the original NumPy equations exactly.
  All 28 source subsets match NumPy FP32 exactly, and FP64 within
  9.175790200699e-07 relative error.
- All outputs and sources are finite; every unfloored variance exceeds
  1e-10. Maximum full-source relative asymmetry is
  1.712759068666e-05,
  within the predeclared 3e-5 tolerance. Native numeric caches retired.

The archived validation losses on 400 and 500 replay within
2.171328543497e-07
relative error. This corroborates the stored model, whose current bytes
were frozen before decoding; the old verdict did not hash-pin the model
and loss equality is not unique parameter-identity proof. The old joint
covariance/response information gate remains failed. This native endpoint
explicitly revised that older stopping decision and has now also failed.

All eight guarded groups retired. The seven full native calls consumed
2,872,944,342,572 measured FLOPs. Peak sampled group RSS was
6,503,514,112 bytes and peak process RSS was
6,503,530,496 bytes. Added swap was zero. About
41.213 MB of artifacts were
preserved before closeout. There was no new fitting, sampling, GPU work,
download, or upload.

Official submission 331684 remains adjusted 3.5844144212297072e-9, raw
1.949715441540434e-8, with 100/100 cases and zero failures. The 4e-9 milestone
is banked; 3e-9, 2e-9, and approximately 1e-9 remain unmet. The full Phase 2
goal remains active.

The separate qualified covariance/readout archive is unchanged. Automatic
approval review rejected its upload before process creation because it could
not verify trusted authorization for the exact private payload and AIcrowd
destination. The existing exact-upload approval question remains unanswered;
this study did not retry or bypass that rejection.
