# Validation-network rounding differences require comparisons against two references

2026-10-01T23:02Z · finding · informational · derived

Every rounding-level variant of the parent (whatever it touched, including a diagnostic with no fast-multiplication rounding at all) changed the final prediction by the same 3.3-5.1e-6 rms and read +0.1..0.2% worse on the 12 screen and check networks the parent's kernels were selected on. A noise decomposition against a plain-matmul noise-free reference showed the parent's rounding noise aligned with its error by -0.149% there (2.2 standard errors) while the variants' alignment was -0.04..+0.06%; on 10 fresh networks the strip stack was -0.107% better. The reading on the selection networks is the parent's selected draw, not a cost.

Recorded measurement:

    prediction change 3.3-5.1e-6 rms (3-4% of the error rms); rounding-noise power RRPI 0.068% of MSE, stack 0.069%, strip builds 0.064-0.074%; alignment RRPI -0.149% (2.2 se; -0.44/-0.45% on 35 and 100), variants -0.04..+0.06%; fresh ranks 660-669: bill -1.863%, MSE -0.107% (8 of 10 lower, mean -0.111%, se 0.069%); validation batch 1: bill -2.113% vs S35, MSE -0.201% (se 0.144%)

## Programs

- [noise_decomp.py](noise_decomp.py)
- [candidate_T_C_B_F_D_Hfull.py](candidate_T_C_B_F_D_Hfull.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L49](../../TEAM_LOG_2026-10-01_to_10-22.md#L49)
- [TEAM_LOG_2026-10-01_to_10-22.md#L56](../../TEAM_LOG_2026-10-01_to_10-22.md#L56)
- [TEAM_LOG_2026-10-01_to_10-22.md#L60](../../TEAM_LOG_2026-10-01_to_10-22.md#L60)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-validation-network-rounding-differences-require-comparisons-against-two) · [Copy provenance](../../../PROVENANCE.json)

