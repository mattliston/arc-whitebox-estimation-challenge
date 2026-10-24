# Termwise Hermite pair-series implementation reduces training intermediates with small rounding differences

2026-09-25T02:58Z · tool · informational · measured

Accumulating the same series term by term avoids broadcast intermediates. Forward and gradient differences were measured before offering it as an optional training acceleration.

Recorded measurement:

    - 2026-09-24 19:58 [engineer] REFIT SPEED DROP-IN (optional, for the refit owners): fable_engineer_v1/pair_series_torch.py (ps.install(card, torch)) replaces the torch card's _pair_series_fused broadcast form with the same Hermite series accumulated term by term. Net 37, S18-physics card: predictions 1.1e-7 relative, final MSE +0.011%; gradients 8.4e-4 median / 1.5e-3 max relative (float32 summation order; the original is exactly deterministic run to run). Alternating timing of the refit step (fine_ckpt, full+P): about -9% +-5% per step after warm-up (31.2->30.1 s, 33.1->28.3 s). Modest; adopt only if the owners want it.

## Programs

- [code/pair_series_torch.py](code/pair_series_torch.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L591](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L591)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../graph/index.html#node=2026-09-25-termwise-hermite-pair-series-implementation-reduces-training-intermediates-small) · [Copy provenance](../../PROVENANCE.json)

