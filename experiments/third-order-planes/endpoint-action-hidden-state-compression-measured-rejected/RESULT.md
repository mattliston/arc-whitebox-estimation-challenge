# Q24 CP3 fixed-rank old-summary discriminator

Status: **FAIL — no rank-2,048 selector is accuracy-viable.**

All three rank-0 arms reserve the exact newest 1,024-column coordinate birth
and use 1,024 columns for an old-state summary.  All entering columns still
contribute to the current K21 reset; dropped post-reset K21 is absorbed into
the dense cubic star.  The native-J logical rank feature is unchanged from the
full-birth host.

| old-summary selector | host final MSE | native-J final MSE | mean q10:15 gap MSE | exact-q10:15 suffix oracle | prototype FLOPs |
|---|---:|---:|---:|---:|---:|
| chronological newest (`w2`, prior receipt) | 4.964476460e-7 | 4.166148553e-7 | 7.640948263e-8 | 8.615974104e-8 | 944,602,732,920 |
| earliest fixed prefix | 6.855265807e-7 | 5.988327805e-7 | 1.098215001e-7 | 1.705079961e-7 | 990,457,311,608 |
| exact symmetric-CP3 self energy | 3.570871516e-7 | 2.502644760e-7 | 4.416530997e-8 | 5.769337540e-8 | 990,458,456,440 |
| R=16 weight-suffix action proxy | 3.662323648e-7 | 2.771014039e-7 | 4.688417679e-8 | 7.266514153e-8 | 991,972,196,728 |
| full growing carrier (prior receipt) | 8.212858909e-8 | 2.856783180e-8 | 5.384855913e-9 | 5.362486676e-9 | 1,574,161,982,840 |

The preregistered rank-0 gate was native-J MSE at most `8.0e-8` and a 2x
improvement over chronological `w2`.  No arm passed.  Energy is the best
selector and improves `w2` by 39.93%, but remains 8.76x worse than the full
carrier.  Earliest-prefix retention is worse than retaining the newest two
cohorts, so old age is not itself valuable.

The action proxy used 16 deterministic Rademacher endpoint probes, reverse
weight products, and the live Q24 marginal sensitivity
`rho3=c1[3]/(6 sigma^3)+c1[6] k3/(36 sigma^6)`.  It captured 81.2--89.9% of its
own score, but its selected set overlapped energy top-k by 79.6--87.6% and
finished 10.72% worse than energy.  The marginal endpoint proxy therefore
does not expose the missing K21/covariance action.

All capped arms first diverge from full one affine layer after their first
selection (zero-based layer 3).  At that layer native-J MSE is `2.053e-9` for
full, versus `7.632e-9` energy, `8.716e-9` action, `9.245e-9` chronological,
and `1.672e-8` prefix.  The immediate failure plus the poor late exact-q
oracles show that the omitted triple-distinct carrier directions are diffuse;
neither age, component energy, nor a cheap marginal suffix action separates a
1,024-column old summary.

Compute conclusion: rank 2,048 was independently identified as the largest
defensible bank (optimistic exact-R40 projection about 549.070B, leaving
26.685B to the unchanged-raw scoring boundary).  Since every equal-rank
selector is far outside the accuracy gate, this compression leaf is closed;
larger-cap follow-up is not justified by the scoring objective.

Frozen bindings:

- prefix/energy runner: `fde24de29896ced87ae31f9ee31a75d5f1863ed7ae17ed8e9972da24039b501e`
- action runner: `e1aec2a7fae1cad0e2ef21379b3adfa787de6d123824ebde1fc553a5e2c2e7cb`
- prefix JSON: `95f7c9f142b8082ee531f7294cc35606e757f4b433cc2fd3c1d8fce9313979f3`
- energy JSON: `490b48835cfcb3b785e65ae9cc39319fd9a3f275cbbec0522b9e05ebe29298b1`
- action JSON: `ebbd8448cb78ea3c381a1fee5816e6f562199312139b1f23ae8e1036f10e6f91`
