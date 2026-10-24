# Global dials near optimum after refit; remaining late-band signals require per-layer constants

2026-10-10T15:10Z · experiment · rejected · measured

The dial re-screen at joint global-dial and learned-array refit (jmH)'s point finds no family off globally any more; only layer 10-14 band sums show |t| about 3, worth -0.2..-0.4% through per-layer literals (late-band per-layer refit (jmI) took iso and cohort-age local fourth-order term (K4L) age 1 per layer).

Recorded measurement:

    largest mean t^2 3.4 (s_carry), bscale 2.8, X1 1.95 (down from 17.1); band 10-14 t: X1 -3.7, iso -3.4, bscale +3.0, k4la1 +3.1

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/cv`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/dial_table.py](code/dial_table.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L655](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L655)
- [TEAM_LOG_2026-10-01_to_10-22.md#L676](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L676)
- [TEAM_LOG_2026-10-01_to_10-22.md#L679](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L679)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [dial_table_0hdg.out](dial_table_0hdg.out) · result

</details>

## Related work

- enables: [Fixed-constant audit: centers and scales are near noise; edge-array gradients fail independent acceptance](../fixed-constant-audit-centers-scales-near-noise-edge-array/README.md)
- enables: [Per-layer fourth-order and isotropic dial refit adopted at layers 10-14](../../../records/research/2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/README.md)
- ← enables: [Refit of four displaced global dials adopted after zero-gradient covariance screens](../../../records/research/2026-10-10-refit-four-displaced-global-dials-adopted-after-zero-gradient/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-global-dials-near-optimum-after-refit-remaining-late-band) · [Copy provenance](../../../PROVENANCE.json)

