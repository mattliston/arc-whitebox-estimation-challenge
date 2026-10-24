# Worst residuals are isolated timing spikes: 1.11-1.17x the run median on otherwise ordinary networks

2026-10-10T02:14Z · finding · informational · measured

The worst network of the graded candidate generation 42 (S42) run (#333607) was network 9 at 0.3561 s residual, not a first network (no setup wall on it); it ran at 1.167x the median of the other 95 networks (0.3051 s) with kernel time per flop only 1.044x the median, the same shape as network 80 of that run (0.3398 s, 1.114x), network 79 of candidate generation 41 (S41)'s run (1.110x) and network 6 of candidate generation 40 (S40)'s (1.114x). The five first networks were 0.3219-0.3313 s. So a safe variant must remove requests from every network, not only first networks.

Recorded measurement:

    network 9 0.3561 s = 1.167x median 0.3051 s, kernel per flop 1.044x; network 80 0.3398 s 1.114x kernel 0.981x; S41 network 79 1.110x; S40 network 6 1.114x; first networks 0.3219-0.3313 s (1.055-1.086x)

## Programs

- [platform_networks.py](platform_networks.py)
- [../2026-10-02-performance-projections-require-latest-graded-parent-fault-free-instruction/new_bank.sh](../2026-10-02-performance-projections-require-latest-graded-parent-fault-free-instruction/new_bank.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L479](../../TEAM_LOG_2026-10-01_to_10-22.md#L479)

<details>
<summary>All result files and records (3)</summary>

- [eval_333607.json](../../../submissions/333607/eval_333607.json) · result
- [new_bank_S42.log](new_bank_S42.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-worst-residuals-isolated-timing-spikes-1-11-1-17x) · [Copy provenance](../../../PROVENANCE.json)

