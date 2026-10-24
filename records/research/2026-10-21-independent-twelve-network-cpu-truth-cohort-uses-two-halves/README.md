# Independent twelve-network CPU truth cohort uses two halves, float64 accumulation and immutable checkpoints

2026-10-21T01:24:59Z · tool · informational · measured

Twelve independent seeds used 64,094,208 samples per network, two truth halves and fixed checkpoints. CPU random-number generation differed from previous GPU bytes but represented the same target distribution; the fresh truths were not fitting targets.

Recorded measurement:

    CPU fresh truth now twelve independent seeds7109001–12,64,094,208 samples each/two halves/f64accumulation/immutable32-batch checkpoints; original v7 guard/shared36/BLAS4. Original8 controller54147 at01:11:31Z; extra4 controller56048 at01:20:37Z after uniqueness audit (prior0096–0103 reused and not fresh). CPU RNG/backend protocol explicitly distinct from MPS bytes, same target distribution; oracle/checkpoint checks passed. Bank777a+edge8 preregistered; never use fresh truth for fits. Latest01:21:44Z (6:21PM Pacific) actual85.92%CPU/12workers/168.84GiBavailable/zero swap/OOM. No100%claim; RAM is mostly free because each truth worker~568MB. Extra4 native-priority admissions as candidates arrive. First8 measured~2.6–2.8h total at original load; not a guarantee under added work.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6440](../../TEAM_LOG_2026-10-01_to_10-22.md#L6440)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-independent-twelve-network-cpu-truth-cohort-uses-two-halves) · [Copy provenance](../../../PROVENANCE.json)

