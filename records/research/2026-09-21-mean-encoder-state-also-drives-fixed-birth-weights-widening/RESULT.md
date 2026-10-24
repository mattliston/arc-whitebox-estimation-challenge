# Current coupling and earlier joint-fit evidence

This audit reads source and existing scalar receipts only. No estimator,
training, numerical runtime, prediction array or target array is executed or
decoded, and no new model or training protocol is selected.

The current fitted P surface is coupled to fixed birth weights. The native
birth-node function consumes the returned GNN state, standardizes its64
channels, multiplies by A[64,4], and applies tanh plus the fixed per-layer
cstar coefficients. The current training translation updates only the11 P
leaves. A, cstar and covariance-edge weights remain frozen, while the source
still feeds the current hs into A(H). No detach is inserted at that use.
This source fact does not certify native gradients or useful optimization.

The64-channel birth-node shape is explicitly checked in native code. Simply
widening the GNN state in its payload would not satisfy that guard or the
A[64,4] interface. Message width is a different dimension: the previous
capacity audit established that it is derived from msg_w and can change
while the state stays64. Neither observation demonstrates compatibility,
cost, resource success or accuracy for an untested model.

Joint P/A learning was actually tested on the older COMP2e13d40a host. Two
matched arms each used20 TRAIN networks,three epochs,60 backwards and15
updates, followed by four sealed CHECK predictions. The joint arm updated
the11 P leaves and256 A entries; the control froze A while retaining its H
dependence. Native CHECK37/39/42/48 compared each role's own complete bills.

Joint/COMP adjusted ratio was1.0009260627183014 with2/4wins; joint/P-only
was1.0044040046045324 with1/4win. The original joint-learning gates failed.
The matched P-only arm's0.9965373078260472 ratio and3/4wins qualified only a
small reused-development component, not native40, independent14, production
or a board result. The score file's top-level passed flag denotes its intact
execution; joint_learning_lead is explicitly false.

Do not conflate this completed joint P/A test with current_bd_p_coadaptation_v1:
the latter was an earlier P-only numerical-qualification attempt with frozen
birth coefficients and stopped before its gradient stages or six-epoch fit.
Also distinguish P/A from the later independent-covariance P/Q pilot, whose
much larger negative result was audited separately. Different names do not
establish different missing experiments; inspect the actual parameter leaves.

These old results do not prove a universal coadaptation limit or forecast the
current fitted candidate. They do rule out calling basic joint P/A tuning an
unexecuted idea. The active retirement-native validation and all original
score gates remain unchanged, with no partial current accuracy preview.
