# Mean-tilt row term double counted the own-unit contribution: correction improves matrix accuracy to 0.96-0.98

2026-10-02T11:44Z · incident · informational · derived

The rebuilt law used (C V)_a summed over all c, but the c = a part is already the own (3,1) term, so it was counted twice; the certified layer 0 to 1 list had excluded it and the check had not been re-run on the rebuilt law. Without the double count the free run from layer 0 holds 0.983 / 0.967 / 0.961 / 0.961 / 0.957 of the true network-specific part at layers 1-5 instead of 0.604 / 0.828 / 0.860 / 0.891 / 0.907; every closed-loop carrier number of the morning was a measurement of the uncorrected law (two beat the Monte Carlo oracle). The corrected full-matrix law gives -7.30% on network 35 (oracle -7.99%) against -6.09% before; the fix went into every carrier and the old term became a dial.

Recorded measurement:

    free run shares layers 1-5: 0.604 / 0.828 / 0.860 / 0.891 / 0.907 as shipped, 0.983 / 0.967 / 0.961 / 0.961 / 0.957 fixed; one step from truth layers 2 / 3 / 5: 0.920 / 0.948 / 0.962 vs 0.969 / 0.973 / 0.971; closed loop 35: certified -7.30% / -8.59% all layers, old own term half -6.65%, full -6.09%, oracle -7.99%; carried share of oracle 0.89 / 0.92 / 0.89 / 0.85 / 0.81 (before 0.76 / 0.88 / 0.87 / 0.84 / 0.80); lean corrected -7.01%; lean rank 32 -5.11%; lean without births -8.32%

## Programs

- [t19u_cvown.py](t19u_cvown.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py)
- [nlaw_v3.py](nlaw_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L270](../../TEAM_LOG_2026-10-01_to_10-22.md#L270)
- [TEAM_LOG_2026-10-01_to_10-22.md#L282](../../TEAM_LOG_2026-10-01_to_10-22.md#L282)
- [TEAM_LOG_2026-10-01_to_10-22.md#L301](../../TEAM_LOG_2026-10-01_to_10-22.md#L301)

<details>
<summary>All result files and records (3)</summary>

- [t19u_cvown_35.log](t19u_cvown_35.log) · log
- [t4_ora17.log](t4_ora17.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-mean-tilt-row-term-double-counted-own-unit-contribution) · [Copy provenance](../../../PROVENANCE.json)

