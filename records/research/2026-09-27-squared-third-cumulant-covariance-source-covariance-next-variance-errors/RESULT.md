# Fixed quadratic covariance-source diagnostic: closed

The source gate fails on both unchanged S31 TRAIN networks. Adding the
previous fixed increment at coefficient1 worsens both pooled error measures:

| TRAIN rank | Off-diagonal covariance error energy | Weighted next-variance error energy | Weighted layer wins |
| --- | ---: | ---: | ---: |
| 35 | +1.234748% | +0.325554% | 4/13 |
| 100 | +1.680668% | +0.455280% | 2/13 |

Every base cross-half energy is positive. The prospective requirements were
at least2% reduction in both pooled measures on both networks, and10/13
weighted layer wins. No sign, scale, subset or refit is selected. This closes
the covariance-alignment rationale for retraining this already failed
candidate. The earlier broader matched-fit failure is also unchanged.

The diagnostic uses existing independent even/odd-batch P32M covariance
halves with exact half sizes16,009,216 and15,990,784 and unbiased denominator
corrections. It observes the actual post-head carried covariance; its delta
never enters inference. Cross-half energies remove independent sampling
variance in exact arithmetic, not shared FP32/MPS bias. These are two reused
TRAIN networks, not independent validation or a population impossibility
claim. No final-mean labels were opened.

All16 prediction files and full operation bills exactly match independent
S31 controls. Four statistic/congruence fixtures pass; source seals and
scalar receipts reconcile. One guard completed and retired in
51.04s, peak RSS5.621GB,
zero added swap. Exec session75412 was reaped with exit0.

No candidate, training, VALID40, sampling, package or upload follows.
Official S31 #332517 remains2.1736465056908115e-9 adjusted. This is a
negative research result and supplies no banked score improvement.
