# All-layer Monte Carlo mean and covariance generator agrees with independent implementation to 6e-9

2026-09-23T03:48Z · tool · informational · measured

The generator records pre- and post-activation moments with independent half-sample noise estimates at all sixteen layers.

Recorded measurement:

    2026-09-22 20:48 covhunter: all-layer MC truth factory certified (fable_covhunter_v1/mc_allayers_mps.py; same stream as fable_mc_v1: L4/7/10/13 agree to 6e-9 rel); mcall_rank{0035,0100}_P32M.npz done (L0-15 pre/post mean+cov, A/B halves; off-diag noise .77% L0 -> .14% L14), r37/r39 running on MPS. Zero-cost knob scan (torch, learned ON, no refit, final MSE vs bank, r35/100/37/39): recentering alpha 0 +3.5% mean (4/4 worse), alpha .5 +0.1% (2/4), off-diagonal recentering removed +3.7%, diagonal removed +0.4%, message edge off +5.1%, edge x.5 +1.7%, response floor .005 +0.4% (2/4), .0025 +1.3%, omitted-radial off +27%. All at a co-adapted optimum; knobs_v1.jsonl.

## Programs

- [code/mc_allayers_mps.py](code/mc_allayers_mps.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L35](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L35)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [knobs_v1.jsonl](knobs_v1.jsonl) · result

</details>

[Research graph](../../graph/index.html#node=2026-09-23-all-layer-monte-carlo-mean-covariance-generator-agrees-independent) · [Copy provenance](../../PROVENANCE.json)

