# Late rank8 congruence covariance write: fixed pilot closed

September5,2026, Phase2 only. No new official submission or milestone.

The target-free action-response certificate passed on35/100, including
byte-identical full-card baseline and both central half-step tests. A fixed
five-epoch20-train/4-held-out warm pilot then trained the current full-card
suffix mean head and a new rank8 PSD-preserving covariance action together.
Training was one MPS job,46.95seconds. No larger fit or sweep was run.

Final actual dynamic NumPy carrier against stored1e9 post means:

| Held-out rank | Parent raw final MSE | New raw final MSE | Ratio |
| --- | ---: | ---: | ---: |
| 37 | 2.15350826147e-8 | 2.17984803517e-8 | 1.01223110 |
| 39 | 1.77858665601e-8 | 1.75264555900e-8 | .98541477 |
| 42 | 2.31795683708e-8 | 2.31580061698e-8 | .99906978 |
| 48 | 2.22270149547e-8 | 2.20795623625e-8 | .99336606 |

All four prefixes through layer9 remain byte-identical. The shadow
predictions were similarly flat; this is not an apparent large training
gain disappearing in deployment. It fails the preregistered8%-on-each-net
gate. Do not run more epochs, learning rates, action scales, ranks, nets,
200-net fits, candidate ports, panels or submissions in this pilot family.

Evidence: late_congruence_fidelity_v1.json,
late_congruence_training_pilot_v1.json, late_congruence_actual_verdict_v1.json;
versioned exported mean/action payloads and four predictions are preserved.
No sealed labels, forum writes, root-estimator edits, commits or deletion.
