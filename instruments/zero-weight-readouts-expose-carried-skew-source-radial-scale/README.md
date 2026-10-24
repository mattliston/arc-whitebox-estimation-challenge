# Zero-weight readouts expose carried skew source, radial scale and plane row gain with unchanged baseline

2026-10-02T19:51Z · tool · informational · measured

Three readouts of learned state are added with zero weights: per-layer carried-skew source, per-layer radial scale and a plane row gain from 52 inputs with constant and layer-linear weights. The certificate preserves baseline predictions and existing learned-array gradients exactly; the proposed post-variance readout is excluded after its oracle ceiling test.

Recorded measurement:

    row-gain inputs52; constant and layer-linear vectors; carried source and radial64 weights/layer; rank3 baseline1.585407264e-8 unchanged in forward and gradient runs

## Programs

- [code/trainer/otr_m.py](code/trainer/otr_m.py)
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
- [code/physicist/pbase_frozen_s4a.py](code/physicist/pbase_frozen_s4a.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L453](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L453)
- [TEAM_LOG_2026-10-01_to_10-22.md#L454](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L454)
- [TEAM_LOG_2026-10-01_to_10-22.md#L459](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L459)
- [TEAM_LOG_2026-10-01_to_10-22.md#L469](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L469)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-02-zero-weight-readouts-expose-carried-skew-source-radial-scale) · [Copy provenance](../../PROVENANCE.json)

