# Dropped star contribution is nearly orthogonal to plane error; optimal scaling removes only 0.1-1.2%

2026-10-11T18:32:18Z · finding · rejected · measured

On networks 35 / 100 of the S47e moment-closure estimator (pinned otr_m, v11 simulator), the star made at layer l and pushed into l+1 is a hub tensor sum_a Sym(w_a, w_a, (P W)_a); its triple-distinct part, transported gate-linearly through l+1 and pushed into l+2, contributes to the l+2 plane a term whose covariance-weighted size is 10-12% of the moment-closure estimator's third-cumulant plane with index multiplicities (2,1) (K21) error at l+2 (8M-sample truth planes, l = 10 / 11 / 12), but the best multiple of it removes only 0.1-1.2% of that error's energy (nearly orthogonal). A one-hop exact carry of the star (about 4 dense products per layer) cannot pay; the structural idea was killed.

Recorded measurement:

    term size 10-12% of the K21 error at l+2 for l = 10 / 11 / 12; best multiple removes 0.1-1.2% of the error energy; the star's full two-layer push is 1.3-1.6x the error's size

## Programs

- [t110_worker_stardump.py](t110_worker_stardump.py)
- [t110_gate.py](t110_gate.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1153](../../TEAM_LOG_2026-10-01_to_10-22.md#L1153)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1154](../../TEAM_LOG_2026-10-01_to_10-22.md#L1154)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-dropped-star-contribution-nearly-orthogonal-plane-error-optimal-scaling) · [Copy provenance](../../../PROVENANCE.json)

