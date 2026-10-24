# Pair-law carrier coefficients shrunk halfway to one: -0.007% adjusted on 32 networks, standard error .136

2026-10-19T05:52:02Z · experiment · rejected · measured

The carrier calibration family theta(s) = 1 + (1-s)(theta0 - 1) with theta0 = (births 1.110, rows 0.693, diag 0.793) of the pair-law carrier (coefficient table coefH rows 1-6 regenerated from the file's own tables) read -0.283% on seven ranks at s = 0.5 but is null on 32 paired ranks: a pure co-adaptation split with per-network sd 0.77 points and zero bill, the shelf's last measured raw lever; the s = 0.25/0.75/1.0 arms were not run.

Recorded measurement:

    7 ranks vs 95c: adjusted -0.283% (se .39; ranks 35/150/200 -0.80%, the next four +0.1%); 32 ranks paired vs the base: all32 -0.007% (se .136, 18/32 lower), heldout25 -0.012% (se .153), seven +0.008%; per-network sd 0.77 points (ranks 150/200/400/500 -0.8..-1.5, ranks 287/293/320/474/623 +1.0..+1.9)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_half_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_half_sci_v1.py](code/candidate_half_sci_v1.py)
- [code/build_t_arms_v1.py](code/build_t_arms_v1.py)
- [code/candidate_s025_sci_v1.py](code/candidate_s025_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5911](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5911)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5913](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5913)

<details>
<summary>All result files and records (2)</summary>

- [science32_half_sci.log](science32_half_sci.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-pair-law-carrier-coefficients-shrunk-halfway-one-0-007) · [Copy provenance](../../../PROVENANCE.json)

