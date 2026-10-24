# Fourth-order input columns show strong gradients; separated skew and transported-source inputs do not

2026-10-02T20:02Z · finding · informational · measured

Zero-weight gradients on 36 networks identify the young-pair fourth-chaos diagonal and local fourth-cumulant value as additional inputs. The split norm-coupling diagonal, transported sources and separated skew contributions have significance comparable to existing trained columns, while the two fourth-order gradients largely align with the already-fitted diagonal input.

Recorded measurement:

    young-pair diagonal 74/128 weights beyond3se, mean t^2 13.38, halves cosine+0.93; local quartic29/128,5.51,+0.82; solved diagonal17/128,3.86,+0.69; other new columns mean t^2 0.44-2.12; three mean-gradient cosines0.89-0.94

## Programs

- [batch2_table.py](batch2_table.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar4.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar4.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar5.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar5.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar6_audit_frozen_s3d.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar6_audit_frozen_s3d.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_fold_frozen_s2w.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_fold_frozen_s2w.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508y.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508y.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1339.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1339.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402x.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402x.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402y.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402y.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L461](../../TEAM_LOG_2026-10-01_to_10-22.md#L461)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-fourth-order-input-columns-show-strong-gradients-separated-skew) · [Copy provenance](../../../PROVENANCE.json)

