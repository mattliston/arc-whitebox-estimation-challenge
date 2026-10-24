# Float32 third-order contractions remain accurate despite 65-81x cancellation

2026-09-23T10:56Z · finding · informational · measured

Direct float32 versus float64 checks found contraction errors far below the multiplication kernel rounding.

Recorded measurement:

    - 2026-09-23 03:56 [engineer] f32 'third' precision (engineer2's cancellation note): in situ r35, float32 vs float64 on the same inputs (f32_precision_check.py): diagonal sum(pp^3 lam) cancels up to 65x, rel error <= 6e-7; omitted-radial contraction cancels up to 81x, rel error <= 1.2e-6 (~4 orders below the kernel's float32 rounding).

## Programs

- [f32_precision_check.py](f32_precision_check.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L182](../../TEAM_LOG_2026-09-22_to_09-26.md#L182)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-float32-third-order-contractions-remain-accurate-despite-65-81x) · [Copy provenance](../../../PROVENANCE.json)

