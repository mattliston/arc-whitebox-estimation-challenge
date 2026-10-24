# Learned joint-graph pilot closed

2026-09-05T15:01Z · experiment · rejected · measured

Fixed24 teacher trajectories and40 epochs completed. Exact host, original features, sigma and frozen-head outputs match the parent. No200-net fit, recursive port, extra epochs/width sweep, panel or submission for this block.

Recorded measurement:

    Fixed24 teacher trajectories and40 epochs completed. Exact host, original features, sigma and frozen-head outputs match the parent. New h/R/T inputs precede current correction. Twenty train nets; held-out37/39/42/48; layers10..15; stored1e9 post means. No target-leak profile failure. Final-epoch held-out mean residual MSE: parent4.80933815e-9, same-capacity local control4.82442897e-9, learned graph4.81360962e-9. Graph/control=.997757, graph/parent=1.000888;0/4 held-out nets improve over parent. This fails the predeclared8%-versus-control/5%-versus-parent gate. Training MSE also improves only slightly (4.85223e-9->4.82451e-9).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/train_learned_joint_graph_probe_v1.py](code/train_learned_joint_graph_probe_v1.py)

## Evidence

- [LEARNED_JOINT_GRAPH_PROBE_V1_CLOSED.md#L1](LEARNED_JOINT_GRAPH_PROBE_V1_CLOSED.md#L1)
- [LEARNED_JOINT_GRAPH_PROBE_V1_CLOSED.md#L12](LEARNED_JOINT_GRAPH_PROBE_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (3)</summary>

- [LEARNED_JOINT_GRAPH_PROBE_V1_CLOSED.md](LEARNED_JOINT_GRAPH_PROBE_V1_CLOSED.md) · record
- [learned_joint_graph_probe_verdict_v1.json](learned_joint_graph_probe_verdict_v1.json) · result
- [LEARNED_JOINT_GRAPH_PROBE_V1.md](LEARNED_JOINT_GRAPH_PROBE_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-learned-joint-graph-pilot-closed) · [Copy provenance](../../../PROVENANCE.json)

