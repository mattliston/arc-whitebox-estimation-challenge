# Apparent limits require independent truth, multiple networks and an instrument audit

2026-09-02 · lesson · informational · measured

Several apparent accuracy limits were caused by incorrect measurement frames or broken instruments. A claimed limit required independent scoring truth, two networks, forcing inside the actual recursion and an audit, while unresolved deficits were to be decomposed into measured mechanisms.

Recorded measurement:

    Matt, 2026-09-01, after I declared the third-order supply "not deployable, no path" and he pushed back: "you are wrong that there is no path … review all work from the last few days … keep chunking blockers into sub problems until you reach a solution, and then recurse back out … to be extremely clear in case you need it - the overarching idea here is to continuously break this problem down into mechanistic sub problems until we solve it."
    
    **Why:** the "no path" verdict came from reading a wedge-cascade artifact as a law (see [[p2-third-order-carrier-wall]]). A blocker stated as a wall hides the mechanism inside it; stated as a mechanism (what error, from which term, how big through the adjoint, what fixes it) it always has children. The repo's whole method is the problem graph: [[graph-nodes-split-and-human-summary]], [[attack-the-blocker-relentlessly]], [[pull-all-threads-before-killing]].
    
    **How to apply:** (1) The root is the scoring metric; each child is a mechanism whose contribution is PRICED through the exact linear integral (the adjoint) against truth — "worth X% of raw". (2) A node with no measured price gets an oracle arm (make that input exact, re-grade) before any building. (3) Recurse into the largest-priced unsolved leaf; a leaf is a formula/carrier whose error can be measured against a truth file. (4) When a leaf is solved, recompose: re-grade the parent, re-price siblings, continue. (5) Never end a turn on "no path" — end on the next leaf and its arm. Living document: `public_g19_math_acceleration_v1/MECHANISM_TREE.md`; graph nodes mirror it.
    
    **Addendum (Matt, 2026-09-01, on the learned F2J operator):** first "preferably you are able to derive everything and do not have to resort to learning", then, after learning F2J is worth 69% of the card's accuracy: "okay you can keep the operator if it helps us get below 1e-8." RULE: derived is preferred; the learned operator STAYS if it helps reach the goal. Its transport is derived already; its learned per-layer source may be kept, refit on an improved card (then [[private-eval-generalization-moat]] gates apply: cross-seed + off-distribution), or replaced by a derived source when that scores at least as well. Never drop it on principle.
    
    **Addendum (Matt, 2026-09-01): "[quote removed]"** Any operator refit runs on the Metal GPU: port the card forward (GEMM-class, ~1.3e12 flops/net) to torch MPS f32 for training-tape generation (the CPU numpy protocol run_joint_refit.py is ~75 s/net × 200 runs ≈ 4.2 h; on MPS seconds/net) and fit in torch. Certify the MPS forward against the numpy card on one net (bit-close, ~1e-6 rel) before trusting tapes. One MPS engine at a time still applies ([[be-gentle-with-the-machine]]).
    
    **Addendum (Matt, 2026-09-02): "[quote removed]"** Standing rule: a "limit" may only be stated after (1) independent-truth scoring, (2) two nets, (3) a frame-free in-card certificate (teacher-force the card itself), and (4) a bug/frame audit of the instrument that produced it. Until all four exist it is a hypothesis under test. Tonight's evidence for the rule: the "tensor wall", the 37-68% post-hoc ceilings, and "k3 is net-specific" were all instrument/frame artifacts; the in-card forcing (T-43) put the card at the noise floor on both nets.
    
    - Matt 2026-09-04 (going to sleep, second time): "you are not permitted to stop. remember that you must keep attacking the blocker until you reach ultimate success. ie keep decomposing whatever the limiting factor is into sub problems until you're able to resolve the entire limiting factor." Current limiting factors (learned-closure program): (1) the autonomous loop's drift (control), (2) the final layer's 29% (information); each has a written decomposition in LANE_g3_tracker T-115/T-116 and an automated attack chain (after_ss05.sh, gate_and_fire.sh).

## Programs

- [../../../experiments/learned-correction/composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py](../../../experiments/learned-correction/composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py)
- [../../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py](../../../experiments/flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/code/public_full688_local_innovation_transport_v1/dataset.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [../../../experiments/learned-correction/composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py](../../../experiments/learned-correction/composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_joint_floor_probe_v1/estimator_joint.py)

## Evidence

- [run_joint_refit.py](../../../experiments/learned-correction/composed-shell-fourth-order-carrier-exported-deterministically-1954-3e9/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_deploy_composition_prep_v1/run_joint_refit.py)

[Research graph](../../../graph/index.html#node=2026-09-02-apparent-limits-require-independent-truth-multiple-networks-instrument-audit) · [Copy provenance](../../../PROVENANCE.json)

