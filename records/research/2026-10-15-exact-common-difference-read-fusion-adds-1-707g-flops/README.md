# Exact common/difference read fusion adds 1.707G FLOPs before packing

2026-10-15T16:29:26Z · finding · rejected · measured

The source cost audit found that fusing the common and difference reads increased arithmetic before any packing cost, closing the proposed exact sharing route.

Recorded measurement:

    adds 1.707G before packing

## Programs

- [price_fused_v1.py](price_fused_v1.py)
- [price_fused_v2.py](price_fused_v2.py)
- [../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [../../../experiments/third-order-planes/read-prefix-cutoff-reduced-1-5-1-0-standard/code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](../../../experiments/third-order-planes/read-prefix-cutoff-reduced-1-5-1-0-standard/code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3425](../../TEAM_LOG_2026-10-01_to_10-22.md#L3425)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [WARM_TARIFF_v1.json](WARM_TARIFF_v1.json) · result
- [WARM_TARIFF_v2.json](WARM_TARIFF_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-exact-common-difference-read-fusion-adds-1-707g-flops) · [Copy provenance](../../../PROVENANCE.json)

