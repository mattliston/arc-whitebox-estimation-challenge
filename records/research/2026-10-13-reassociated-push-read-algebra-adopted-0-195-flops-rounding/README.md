# Reassociated push/read algebra adopted: -0.195% FLOPs at rounding-level prediction change

2026-10-13T02:14Z · decision · adopted · measured

Astra's independent hunt found a reassociation of the push / read algebra worth -0.195% of metered FLOPs with a prediction change bounded at 0.054% of MSE; its rounding-conditioning variants (BLcancel, rightscale, final64) move further from the float32 reference, so the drift is a re-roll of reassociated products, not a cancellation site; HZT (-0.018%) and ASZ were byte-identical add-ons.

Recorded measurement:

    7100136: bill -0.1947%, requests -188, |d|^2 / MSE 0.0542%, actual MSE +0.134%; core ablation -0.182% bill but MSE +0.44%; HZT -0.018004% byte-identical; BLcancel 3.30e-6 vs 3.10e-6 reference drift

## Programs

- [candidate.py](candidate.py)
- [candidate-357ef9bd.py](candidate-357ef9bd.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1863](../../TEAM_LOG_2026-10-01_to_10-22.md#L1863)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1870](../../TEAM_LOG_2026-10-01_to_10-22.md#L1870)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1880](../../TEAM_LOG_2026-10-01_to_10-22.md#L1880)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1881](../../TEAM_LOG_2026-10-01_to_10-22.md#L1881)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1887](../../TEAM_LOG_2026-10-01_to_10-22.md#L1887)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1922](../../TEAM_LOG_2026-10-01_to_10-22.md#L1922)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1932](../../TEAM_LOG_2026-10-01_to_10-22.md#L1932)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1933](../../TEAM_LOG_2026-10-01_to_10-22.md#L1933)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1991](../../TEAM_LOG_2026-10-01_to_10-22.md#L1991)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1992](../../TEAM_LOG_2026-10-01_to_10-22.md#L1992)

<details>
<summary>All result files and records (6)</summary>

- [native_stack_v2_cert_comparison.json](native_stack_v2_cert_comparison.json) · result
- [native_core_v1_cert_comparison.json](native_core_v1_cert_comparison.json) · result
- [hzt_native_v1_cert_comparison.json](hzt_native_v1_cert_comparison.json) · result
- [REPORT.md](REPORT.md) · explainer
- [OPUS_HANDOFF_20261013.txt](OPUS_HANDOFF_20261013.txt) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [FLOP census: two Winograd leaf products consume 67%; float64-rate work is only 0.56%](../../../experiments/flop-accounting/flop-census-two-winograd-leaf-products-consume-67-float64/README.md)
- enables: [Exact compute-copy and retained-tree changes measured: -0.0533768% to -0.1328245% FLOPs](../2026-10-13-exact-compute-copy-retained-tree-changes-measured-0-0533768/README.md)
- shipped in: [Submission #335696: reassociated native algebra and identity caching](../../../submissions/335696/README.md)
- shipped in: [Submission #335709: final-layer hub quartic ages 3-4 and byte-identical product cuts](../../../submissions/335709/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-reassociated-push-read-algebra-adopted-0-195-flops-rounding) · [Copy provenance](../../../PROVENANCE.json)

