# Replaying carrier additions as constants restores gradient runs: finite-difference control correlation 1.00

2026-10-02T15:30Z · tool · informational · measured

The carrier modules keep their state outside the autograd graph and the per-layer checkpoints recompute every step inside the backward pass, so a gradient run with a module installed had been invalid. A replayed step now takes what the modules added in the forward pass (the fold term of the bank read, the extra plane, the (2,2) fourth-cumulant plane (K22) change) from a record and does not call them: the carrier is a constant of each layer, the card's arrays and dials get gradients lacking only the paths through the carrier's own dependence on the state, and the carrier's dials are solved from forward runs. Certified on 16 line-search ranks with a positive control against forward dial arms.

Recorded measurement:

    forward error of a gradient run equals the forward-only run on all 16 networks (largest difference 0); replayed K21 plane equals the forward pass to 1e-6; control c0 -0.1: predicted +0.43%, forward +0.86% (se 0.22), per-network correlation 1.00; k31 knob -0.2: +0.59% predicted, +1.43% real (se 0.30), correlation 0.99; gradient run 56 s in the pool

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L359](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L359)
- [TEAM_LOG_2026-10-01_to_10-22.md#L377](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L377)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-02-replaying-carrier-additions-constants-restores-gradient-runs-finite-difference) · [Copy provenance](../../PROVENANCE.json)

