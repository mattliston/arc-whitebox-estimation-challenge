# Zero-weight input columns test new information without changing the learned correction at baseline

2026-10-02T17:53Z · tool · informational · measured

Each new standardized per-unit input enters the first layer as a rank-one bias term with 128 zero weights. The unchanged baseline and unchanged existing gradients are certified, and finite differences verify derivatives before a 36-network significance table selects inputs.

Recorded measurement:

    126 original inputs = 30 features + 96 messages; 128 weights per candidate column; expected 0.35 weights beyond 3 standard errors under null; mean t squared null 1.0, sd 0.125; step 0.02 measured -0.526% vs first-order -0.528%; step 0.05 -1.220% vs -1.321%

## Programs

- [code/trainer/otr_m.py](code/trainer/otr_m.py)
- [code/trainer/cert/law_ls/head_input_table.py](code/trainer/cert/law_ls/head_input_table.py)
- [code/ncar.py](code/ncar.py)
- [code/ncar4.py](code/ncar4.py)
- [code/ncar5.py](code/ncar5.py)
- [code/physicist/b6cols.py](code/physicist/b6cols.py)
- [code/physicist/b7cols.py](code/physicist/b7cols.py)
- [code/physicist/f4b.py](code/physicist/f4b.py)
- [code/physicist/ncar6_audit_frozen_s3d.py](code/physicist/ncar6_audit_frozen_s3d.py)
- [code/physicist/ncar_bank_frozen_s2s.py](code/physicist/ncar_bank_frozen_s2s.py)
- [code/physicist/ncar_fold_frozen_s2w.py](code/physicist/ncar_fold_frozen_s2w.py)
- [code/physicist/ncarry.py](code/physicist/ncarry.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/physicist/pbase_frozen_s4a.py](code/physicist/pbase_frozen_s4a.py)
- [code/trainer/frozen/k31l_t1508.py](code/trainer/frozen/k31l_t1508.py)
- [code/trainer/frozen/k31l_t1508y.py](code/trainer/frozen/k31l_t1508y.py)
- [code/trainer/frozen/ncar6_t1339.py](code/trainer/frozen/ncar6_t1339.py)
- [code/trainer/frozen/ncar6_t1402.py](code/trainer/frozen/ncar6_t1402.py)
- [code/trainer/frozen/ncar6_t1402x.py](code/trainer/frozen/ncar6_t1402x.py)
- [code/trainer/frozen/ncar6_t1402y.py](code/trainer/frozen/ncar6_t1402y.py)
- [code/trainer/otr_common.py](code/trainer/otr_common.py)
- [code/trainer/otr_pl.py](code/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L406](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L406)
- [TEAM_LOG_2026-10-01_to_10-22.md#L407](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L407)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-02-zero-weight-input-columns-test-new-information-without-changing) · [Copy provenance](../../PROVENANCE.json)

