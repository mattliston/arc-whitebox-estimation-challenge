# Mixed rectangular multiplication: optimistic envelope, not a submission

2026-09-06T00:53Z · experiment · informational · measured

A bounded dynamic program mixes these six shapes, current packed Winograd, and the published accurate rational 4x4 rank48 arithmetic counts. Maximum five recursion stages, minimum leaf dimension eight. All six cyclic/transpose versions of the primary Smirnov 3x3x6 rank40 tensor passed exact rational basis-product certificates.

Recorded measurement:

    A bounded dynamic program mixes these six shapes, current packed Winograd, and the published accurate rational 4x4 rank48 arithmetic counts. Maximum five recursion stages, minimum leaf dimension eight. It uses the existing byte-exact bank matmul census for ranks35/100; no new estimator run. Pad-only v1 offers optimistic whole-card savings18.043%/18.484%. Version2 adds exact classical fringe peeling at every stage, with inner-fringe accumulation additions included, giving19.7082833%/19.9162497%. At the unchanged official raw score, the4e-9 threshold needs19.7146895% savings. Even the optimistic envelope therefore only brushes that threshold. Requiring necessary full-batch leaf A/B/C storage to fit the current arena reduces the v2 envelope to17.5137576%/17.5853781%. This remains only a memory lower bound, not complete live storage. New-algorithm copies, allocations, remote overhead and full scheduling memory are NOT charged. No practical saving, runtime gain, raw score or threshold bank is claimed. The broad scalar integer-power receipt was also re-read: relative to the already-selected CP-cube rewrite it saves only13.787M FLOPs on rank35, while worsening remote wall85.10s to94.27s and residual .36836s to .41537s. It is not an untested several-percent cost lever.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/audit_smirnov_cyclic_v1.py](code/audit_smirnov_cyclic_v1.py)
- [code/audit_smirnov_slp_v1.py](code/audit_smirnov_slp_v1.py)
- [code/price_mixed_fmm_envelope_v1.py](code/price_mixed_fmm_envelope_v1.py)
- [code/price_mixed_fmm_peel_v2.py](code/price_mixed_fmm_peel_v2.py)
- [code/price_smirnov_executable_transpose_v1.py](code/price_smirnov_executable_transpose_v1.py)
- [code/price_smirnov_linear_circuit_v1.py](code/price_smirnov_linear_circuit_v1.py)

## Evidence

- [MIXED_FMM_ENVELOPE_V2_VERDICT.md#L1](MIXED_FMM_ENVELOPE_V2_VERDICT.md#L1)
- [MIXED_FMM_ENVELOPE_V2_VERDICT.md#L17](MIXED_FMM_ENVELOPE_V2_VERDICT.md#L17)
- [MIXED_FMM_ENVELOPE_V2_VERDICT.md#L22](MIXED_FMM_ENVELOPE_V2_VERDICT.md#L22)

<details>
<summary>All result files and records (4)</summary>

- [MIXED_FMM_ENVELOPE_V2_VERDICT.md](MIXED_FMM_ENVELOPE_V2_VERDICT.md) · record
- [smirnov_cyclic_certificate_v1.json](smirnov_cyclic_certificate_v1.json) · result
- [mixed_fmm_envelope_v1.json](mixed_fmm_envelope_v1.json) · result
- [mixed_fmm_envelope_peel_v2.json](mixed_fmm_envelope_peel_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-mixed-rectangular-multiplication-optimistic-envelope-not-submission) · [Copy provenance](../../../PROVENANCE.json)

