# Native route smoke tests: doubled pair skew/kurtosis adds 2.450% MSE; kink scale +0.5 adds 147.946%

2026-10-13T03:38Z · finding · informational · measured

The non-empty native paths of the new refit families were smoke-tested on the submitted twin at network 7100264: doubling the pair map's local skew and kurtosis terms at layer 12 and a kink third-cumulant plane with index multiplicities (2,1) (K21) scale of +0.5 at layer 12 both leave layers 0-11 bit for bit and change the output from layer 13 on (layer 12 itself moves slightly, 1.9e-6 and 1.6e-7, through its learned correction); the kink K21 scale is two orders of magnitude more sensitive. The five PyTorch parities (corr 0.998-1.000, D 0.995-1.000) confirmed the routes end to end.

Recorded measurement:

    pe34 e3 / e4 x2 at L12: relative change L13 1.6e-05, L14 2.2e-05, L15 2.4e-05, final MSE +2.450%; k21k +0.5 at L12: L13 1.1e-04, L14 1.5e-04, L15 1.8e-04, final MSE +147.946%; layers 0-11 bit for bit; L12 itself 1.9e-6 (pe34 x2) and 1.6e-7 (k21k +0.5)

## Programs

- [smoke_pe34k21k_v1.sh](smoke_pe34k21k_v1.sh)
- [../../../instruments/native-routes-add-pair-map-skew-kurtosis-kink-plane/code/opus_reexam_20261001_v1/engineer/a62_modules/make_pe34_v1.py](../../../instruments/native-routes-add-pair-map-skew-kurtosis-kink-plane/code/opus_reexam_20261001_v1/engineer/a62_modules/make_pe34_v1.py)
- [../../../instruments/native-routes-add-pair-map-skew-kurtosis-kink-plane/code/opus_reexam_20261001_v1/engineer/a62_modules/make_k21k_v1.py](../../../instruments/native-routes-add-pair-map-skew-kurtosis-kink-plane/code/opus_reexam_20261001_v1/engineer/a62_modules/make_k21k_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py](../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1913](../../TEAM_LOG_2026-10-01_to_10-22.md#L1913)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1997](../../TEAM_LOG_2026-10-01_to_10-22.md#L1997)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-native-route-smoke-tests-doubled-pair-skew-kurtosis-adds) · [Copy provenance](../../../PROVENANCE.json)

