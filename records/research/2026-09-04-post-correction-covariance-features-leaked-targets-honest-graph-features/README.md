# Post-correction covariance features leaked targets; honest graph features improved only 3 percentage points

2026-09-04 · incident · failed · measured

The apparent 97.8% held-out gain came from post-correction covariance features that encoded the target. Recomputing these features on the estimator's own covariance reduced held-out removal to 67.7% versus 64.3% without them, leaving a real but small graph-feature gain.

Recorded measurement:

    - T-137 (running): Puffi-style CORRELATION-GRAPH features for the closure (puffi_feats.py: per unit, the 8 strongest |R-| neighbours' weighted sums of (a, g, phi), sum r^2, max |r|, softmax-gamma neighbour means (gamma 8, 32), post top-8 eigen loading pi_i, <R+>, <R+^2>, log-var, (C+a)/s, (C+g)/s = 20 cols) computed from a torch replication of the honest Gaussian-card oracle recursion (matches the row host to 2e-6 at L8); trained with the 48-msg closure (GNN_COVFEAT=1 GNN_COVFEAT_FILE=puffifeat.npy) vs the 64.3% baseline. Rationale: Puffi's rank-4 phase-2 card uses a tiny third-order carrier + a trajectory-fitted per-unit corrector whose inputs aggregate over the correlation graph and the top-8 covariance subspace (the organizers' hint in feature form); our closure's inputs never had those.
    
    - T-137 FIRST RESULT = SUSPECTED LEAK: with the correlation-graph features the closure removed 97.8% held-out (final 1.31e-8 vs host 5.9e-7; L8/12/14 96/97/97%) vs 64.3% without. BUT the post-block features (pi, <R+>, <R+^2>, log var, C+a, C+g) were computed from Cn = Cp - m d^T - d m^T - d d^T, the covariance AFTER the oracle correction, which contains the target d (log(Cn_ii) leaks d_i). Same trap as T-116g. Fix (T-137b): post block from the card's own Cp; regenerating (puffifeat2.npy) and retraining. If the removal stays far above 64%, the features are real; if it falls back, it was the leak.
    
    - T-137b RESULT: leak-fixed correlation-graph features (post block from the card's own covariance): held-out removal 67.7% peak (epoch 20) vs 64.3% without; L8/12/14 65/65/67. The 97.8% was ENTIRELY the leak (post-correction covariance encodes the target). +3 points is real but small: the per-unit closure ceiling (~2/3) stands even with cross-unit correlation-graph inputs. LEAD DEAD as a 2x lever. Second target-leak trap of the project: any feature computed from a state that was updated with the correction (mean OR covariance) leaks; the checklist now covers the covariance.

## Programs

- [puffi_feats.py](puffi_feats.py)
- [../../../experiments/flop-accounting/differentiable-gaussian-closure-four-compensator-designs-stopped-6668-6/code/diff_card.py](../../../experiments/flop-accounting/differentiable-gaussian-closure-four-compensator-designs-stopped-6668-6/code/diff_card.py)

## Evidence

- [LANE_g3_tracker.md#L947](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L947)
- [LANE_g3_tracker.md#L949](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L949)
- [LANE_g3_tracker.md#L952](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L952)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-post-correction-covariance-features-leaked-targets-honest-graph-features) · [Copy provenance](../../../PROVENANCE.json)

