# Learned joint-graph pilot closed

Fixed24 teacher trajectories and40 epochs completed. Exact host, original
features, sigma and frozen-head outputs match the parent. New h/R/T inputs
precede current correction. Twenty train nets; held-out37/39/42/48;
layers10..15; stored1e9 post means. No target-leak profile failure.

Final-epoch held-out mean residual MSE: parent4.80933815e-9,
same-capacity local control4.82442897e-9, learned graph4.81360962e-9.
Graph/control=.997757, graph/parent=1.000888;0/4 held-out nets improve
over parent. This fails the predeclared8%-versus-control/5%-versus-parent
gate. Training MSE also improves only slightly (4.85223e-9->4.82451e-9).

No200-net fit, recursive port, extra epochs/width sweep, panel or submission
for this block. Receipt: learned_joint_graph_probe_verdict_v1.json.
This rejects the specified learned one-hop R/R2/T/T-transpose messages,
not a universal statement about all neural or joint-structure estimators.
