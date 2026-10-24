# Old and young read scales, core scale and star coefficient: small perturbations do not improve error

2026-10-19T06:45:07Z · experiment · rejected · measured

Finite differences on the fired bundle's read-scale dials: the old-tier read scale at layers 7-14 is a sensitive dial (5% moves cost 1.5-1.6% MSE) with a parabola optimum at 0.9985, the Nystrom core rescale is insensitive, the young read scale is a sharp V, the old-tier scale at layers 8-10 is flat, and star beta 0.85 beats 0.80 and 0.90: nothing to re-center. Six of the arms first failed because the judge's 7.5 GB RSS backstop retired them at rank 200, not on physics.

Recorded measurement:

    old read scale x0.95 at L7-14: raw +1.550% (se .73, 1/7 lower), bill -0.14%; x1.05: +1.604% adjusted vs 95c (+1.85% vs #337028); parabola optimum 0.9985; core rescale x0.95: -0.270% vs 95c = -0.02% vs #337028 (se .10); young read scale x0.97 raw +2.61% / x1.03 +4.02%; old-tier L8-10 x0.98 +0.03% / x1.02 -0.03%; star beta 0.80 +0.13% / 0.90 +0.18% (0.85 optimum); all 7 ranks vs 95c

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_oct19_coadapt_v1/candidate_jsym_rso095_v1.py](code/fable_oct19_coadapt_v1/candidate_jsym_rso095_v1.py)
- [code/fable_oct19_coadapt_v1/candidate_jsym_rso105_v1.py](code/fable_oct19_coadapt_v1/candidate_jsym_rso105_v1.py)
- [code/fable_oct19_coadapt_v1/candidate_jsym_core095_v1.py](code/fable_oct19_coadapt_v1/candidate_jsym_core095_v1.py)
- [code/fable_oct19_dials_v1/candidate_rsy097_sci_v1.py](code/fable_oct19_dials_v1/candidate_rsy097_sci_v1.py)
- [code/fable_oct19_dials_v1/candidate_rsy103_sci_v1.py](code/fable_oct19_dials_v1/candidate_rsy103_sci_v1.py)
- [code/fable_oct19_dials_v1/candidate_beta080_sci_v1.py](code/fable_oct19_dials_v1/candidate_beta080_sci_v1.py)
- [code/fable_oct19_dials_v1/candidate_beta090_sci_v1.py](code/fable_oct19_dials_v1/candidate_beta090_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5901](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5901)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5902](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5902)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5915](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5915)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5919](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5919)

<details>
<summary>All result files and records (3)</summary>

- [run_judges_v1.log](run_judges_v1.log) · log
- [run_pairs_rss79.log](run_pairs_rss79.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-old-young-read-scales-core-scale-star-coefficient-small) · [Copy provenance](../../../PROVENANCE.json)

