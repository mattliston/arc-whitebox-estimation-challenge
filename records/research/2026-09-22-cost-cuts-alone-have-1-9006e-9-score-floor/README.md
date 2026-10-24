# Cost cuts alone have a 1.9006e-9 score floor at the measured error of submission #331824

2026-09-22T16:30Z · finding · informational · derived

A research assessment written for the ~1e-9 goal derives the reachable scores from the per-network formula E_i x max(0.1, F_i / 2^41): with predictions unchanged the floor is 1.9006e-9, 1e-9 needs at least a 47.38% raw-error reduction (71.49% at unchanged costs), and a uniform 14.459% cost cut reaches 3e-9 with every bill still above the floor. It also records that adding dF flops to a network needs a fractional raw improvement above dF/(F+dF) to break even, and ranks five research levers (co-adapted moments, the joint remainder, distillation, native-objective agreement, cheap bankable gains).

Recorded measurement:

    floor with identical predictions 1.9005926184e-9; 1e-9 needs raw -47.38% (uniform, at the floor) or -71.49% (unchanged costs); 3e-9 at a uniform 14.459% cost cut; the readout's 9.404 GF needs 2.28% raw; +45 / +100 GF need 10.06% / 19.91%; bank bill 381.025-425.364 GF per network (mean 402.386)

## Evidence

- [RESEARCH.md#L15](RESEARCH.md#L15)
- [RESEARCH.md#L16](RESEARCH.md#L16)

<details>
<summary>All result files and records (3)</summary>

- [RESEARCH.md](RESEARCH.md) · record
- [economics.json](economics.json) · result
- [goal_research_checkpoint154.md](goal_research_checkpoint154.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-cost-cuts-alone-have-1-9006e-9-score-floor) · [Copy provenance](../../../PROVENANCE.json)

