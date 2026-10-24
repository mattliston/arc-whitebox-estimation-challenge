# Per-network gradient records provide band and layer sensitivities on 64 generated networks

2026-10-13T01:57Z · tool · informational · measured

The refit rounds' gradient runs were re-read into per-network records: for each of the 64 A36 seeds the final MSE, per-layer MSEs, band-summed gradients (L5-9 / L10-11 / L12-14 / L15 / all) for every per-layer family, the 16 per-layer gradients of 52 families and the global dials' gradients, in units d(1e8 x final MSE)/d(dial), for candidate generation 53 (S53) (transport gate on) and candidate generation 52 (S52) (gate off) at the same start point. These records fed the kink-feature correlation screens and the kink-gated per-layer learned-mean-correction scale (ds) test.

Recorded measurement:

    64 seeds; 52 per-layer families; bands L5-9 / L10-11 / L12-14 / L15 / all; records mix 8-thread and 2-thread workers (rounding level only)

## Programs

- [code/pernet_grads_v1.py](code/pernet_grads_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1873](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1873)

<details>
<summary>All result files and records (3)</summary>

- [pernet_grads_v1.json](../../experiments/dials/kink-dead-unit-top-mode-features-fail-predict-per/pernet_grads_v1.json) · result
- [pernet_grads_v1.json](pernet_grads_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-13-per-network-gradient-records-provide-band-layer-sensitivities-64) · [Copy provenance](../../PROVENANCE.json)

