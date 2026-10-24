# Marginal Edgeworth transport gate raises pooled adjusted error to 1.007519821658286

2026-09-19 · experiment · rejected · measured

A clipped first-order non-Gaussian gate modified inherited cubic-factor transport; the later scope review distinguishes its coupled birth effect from isolated history transport.

Recorded measurement:

    Phase 2 only. The fixed correction passed its algebra and native execution
    development gate. Weight5 remains the best qualified local candidate.
    `a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9`.
    `decef858c58303525f93da024f85a475b6b4f27b87595b19b139904ba7ac13e5`.
    With alpha = mu/sigma, gamma3 = k3/sigma^3 and gamma4 = k4/sigma^4:
             [gamma3*(alpha^2-1)/6 - gamma4*(alpha^3-3*alpha)/24]
        gCP = clip(gE, 0, 1)
    restores the target repeated plane in real arithmetic. Original FP32 factor
    storage and FP64 plane assembly remain; this is not a claim of bit-exact
    Independent 192-node Gauss-Legendre integration of the explicit Hermite
    density passes all 44 fixed fixtures: maximum absolute error
    2.6367796834847468e-14 against the 2e-12 limit. Dense three-leg gating,
    2.220446049250313e-16. Zero cumulants recover the Gaussian gate exactly.
    Full native 35/100/35 and independent cold100 produce finite, nonnegative
    FP32 outputs at all 16 layers. Own repeat, warm/cold100, and the first two
    served layers versus Weight5 are byte exact. Outputs, source, weights and
    complete bills were sealed before opening the existing 1B reference means.
    | Development network | Weight5 raw MSE | Candidate raw MSE | Candidate own-cold FLOPs | Candidate adjusted | Adjusted / Weight5 |
    | 35 | 1.871255439571699e-8 | 1.8829235314153358e-8 | 405548334015 | 3.4725258103350366e-9 | 1.0062368833100865 |
    | 100 | 1.854682334048165e-8 | 1.871009454023561e-8 | 408576506924 | 3.4763184300878806e-9 | 1.0088046306080374 |
    The pooled adjusted ratio is **1.007519821658286**, a 0.751982% worsening.
    Raw ratios are 1.0062354350971492 and 1.008803189460353. Both individual
    improvement requirements and the pooled <=0.99 requirement fail. These are
    two reused development networks with ordinary 1B errors, not the 14-network
    Four complete prediction calls total 1,594,369,321,838 billed FLOPs. Each costs

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/gate_transport_v1/build.py](code/gate_transport_v1/build.py)
- [code/gate_transport_v1/candidate.py](code/gate_transport_v1/candidate.py)
- [code/gate_transport_v1/kernel.py](code/gate_transport_v1/kernel.py)
- [code/gate_transport_v1/score_dev.py](code/gate_transport_v1/score_dev.py)
- [code/gate_transport_v1/unit.py](code/gate_transport_v1/unit.py)
- [code/weight5_reuse_v1/candidate.py](code/weight5_reuse_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L12](NEGATIVE_RESULT.md#L12)
- [REEXAMINATION_INDEX_20260919.md#L1405](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1405)
- [REEXAMINATION_INDEX_20260919.md#L1407](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L1407)

<details>
<summary>All result files and records (5)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-marginal-edgeworth-transport-gate-raises-pooled-adjusted-error-1) · [Copy provenance](../../../PROVENANCE.json)

