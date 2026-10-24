# On-policy first-128 extension is unobservable from captures lacking source-specific full hidden features

2026-10-21T08:20:42Z · finding · informational · measured

The existing hook stored insufficient features to reconstruct the current first-hidden trajectory, and SiLU was noninjective. Reusing older trajectory features or guessed inverse branches was rejected before any reconstruction or fit.

Recorded measurement:

    OP/first128_extension_assessment_v1/OBSERVABILITY_CLOSURE_v1.json 9da7a71738e5a2b9b7beaad03050b4b04165aa317660270b5ee8b9d19125c61b. Frozen hookfad80fe6 stores state64/hidden2/sigma/delta/H[:,5], not H78/first128/dynamicHCbias. Certifiedfirst reconstructiona88658a9 needs source-specificH78; available777a H belongs to a different trajectory. Unchangedenc2 cannot resolve noninjectiveSiLU branches. Rootagreed tohold/close this no-new-capture extension; no guessedbranches, globalcoefficienttransfer, or777a H reuse. No feature/target arrays opened or reconstruction/fit attempted. Original on-policy pipeline and its hidden-null64 secondstep unchanged. A new capture would require a separate value/resourceprotocol, not automaticreplay.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6993](../../TEAM_LOG_2026-10-01_to_10-22.md#L6993)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [OBSERVABILITY_CLOSURE_v1.json](OBSERVABILITY_CLOSURE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-policy-first-128-extension-unobservable-captures-lacking-source-specific) · [Copy provenance](../../../PROVENANCE.json)

