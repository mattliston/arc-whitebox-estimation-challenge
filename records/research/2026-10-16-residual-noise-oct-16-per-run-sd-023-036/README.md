# Residual noise on Oct 16: per-run sd .023 to .036 and spikes of +10-28% on 1-6 networks; P(any fail) 89% at median .355

2026-10-16T14:44:42Z · finding · informational · measured

A log model (run + network + noise) over 22 graded runs found that the estimator submitted as #336412 (S2tr) per-network residual equals the earlier clean builds (median ratio .964 vs #336342, 1.003 vs #336334) and that every Oct 16 run had 1-6 networks spiking +10-28% (noise sd .023 -> .036, q99.9 +.107 -> +.248); wall noise also rose (sd .024 -> .040). This was first read as a noisier platform and set the clean-gate level (warm <= 41,200 requests, median <= .32 s). The reading was corrected at 22:22Z: the spikes are participant-side (lazy view binds vary 13-72 per network) plus worker-speed episodes, not a platform regime.

Recorded measurement:

    Noise sd .023 -> .036, q99.9 +.107 -> +.248; P(any/public fail) at run median .355: 89/70%, .32: 28/17%, .31: 14/8.5%, .30: 5/3% (Oct 13-15 noise: .33 ~0%). Wall sd .024 -> .040, q99.9 +.215; S2tr platform wall median 96.6 s -> P(any > 120 s) ~15-18% under the Oct 16 noise, ~0% at median <= 90 s.

## Programs

- [../2026-10-16-dynamic-residual-routing-dropped-residual-time-optimised-statically-under/count_gates_v2.py](../2026-10-16-dynamic-residual-routing-dropped-residual-time-optimised-statically-under/count_gates_v2.py)
- [resid_compare_v1.py](resid_compare_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4249](../../TEAM_LOG_2026-10-01_to_10-22.md#L4249)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4251](../../TEAM_LOG_2026-10-01_to_10-22.md#L4251)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4254](../../TEAM_LOG_2026-10-01_to_10-22.md#L4254)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4260](../../TEAM_LOG_2026-10-01_to_10-22.md#L4260)

<details>
<summary>All result files and records (2)</summary>

- [eval_336412.json](../../../submissions/336412/eval_336412.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- supersedes: [Paired telemetry review: the lost networks are participant-side spikes and slow-worker episodes, not a platform shift](../2026-10-16-paired-telemetry-review-lost-networks-participant-side-spikes-slow/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-residual-noise-oct-16-per-run-sd-023-036) · [Copy provenance](../../../PROVENANCE.json)

