# Frozen learned endpoint histories as a covariance decoder

September6,2026, Phase2 only. This is a new information test, no state write.
The earlier315-coefficient covariance kernel used14 local analytic response
features. The late PSD-congruence pilot used rank8 covariance actions trained
through final loss. The learned graph-head pilot produced scalar means.
None tested a FULL-RANK covariance kernel decoded from BOTH endpoints'
learned recurrent histories on the actual coherent covariance-source labels.
Those prior rounds remain closed. This is not a retry of their optimizers.

Use the frozen bank P0 GNN as a target-free sidecar on the existing coherent
8M teacher trajectories. State H_l comes from the current pre-correction
30 features and prior P0 state, never current true mean/covariance feedback.
Appending a constant gives65 endpoint features. Fix R=pre correlation.
Decoder E_ij = sigma_i sigma_j sum_{r=0..2} R_ij^r H_i^T Theta_r H_j,
three symmetric65x65 matrices SHARED across layers. The R/R^2 products
need not be rank16: this differs from the old low-rank congruence. It does
not prove new information independent of nonlinear model capacity; the
test asks whether this particular deployable history decoder is useful.

Prerequisite: regenerate only the eight sidecar H arrays while replaying
the EXACT coherent8M-v2 teacher. Require byte equality for all archived
basis/rho/sigma/target/weight/response arrays on every net/layer, no relaxed
boundary. Save only H and receipts; reuse existing target/operator arrays,
not redundant large files. Frozen P0 weights/source hashes recorded.

Same six train35/100/150/200/250/300, held-out400/500, uniform layers8..14.
Normalize H channels by TRAIN-only RMS, min1e-6; sigma by train RMS. No
whitening, basis truncation, rank search or extra features. All Theta zero
initialization; exactly100 Adam epochs LR.02 cosine to.002, seed5290722,
ridge1e-3. Balance covariance and next-mean-response MSE by their zero
predictor TRAIN energies, identical to the old fixed-kernel objective.
Final epoch only; no validation-based checkpoint selection.

At final epoch require covariance AND next-mean-response error ratios
<=.75 to zero decoder on EACH held-out400/500. Also report old14-feature
kernel comparison, but do not repurpose that weaker baseline as the gate.
Failure closes this history decoder before any covariance write or fit
expansion. PASS would still need independent-noise, on-policy exposure,
PSD/stability, matched heads, native cost,14-panel and persistent runtime
gates. No new labels/MC, no forum post, no submission under this design.
