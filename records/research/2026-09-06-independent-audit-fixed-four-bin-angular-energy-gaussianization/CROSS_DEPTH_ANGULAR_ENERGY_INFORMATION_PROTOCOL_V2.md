# Cross-depth angular-energy conditioning: protocol v2 gate amendment

Phase 2 only, September 6, 2026. **Written before acquisition or new target
access, at root's explicit design-review instruction. No acquisition is
authorized yet.** Preserve v1 as the original draft; do not modify it.

This document plus sections 1--5 and 8 of
`CROSS_DEPTH_ANGULAR_ENERGY_INFORMATION_PROTOCOL_V1.md` are the prospective
protocol. **This v2 replaces v1 sections 6 and 7 completely.** The
third-moment information ratio is now diagnostic only. Higher cumulants
can compensate a third-moment projection, so a failed third ratio is not
a necessary condition for failure of the literal Gaussian-mixture mean.
This change is made before any new samples, target-array reads, results,
or target-dependent choices. The tiny certificate source is unchanged.

## Fixed question and acquisition, unchanged

One network only: observer-fit rank 35, seed 2758000291903877946. Zero-based
post-ReLU hidden activations H_4 and H_8, preactivation Z=H_12 W_13.

    A_l = 1024 ||H_l(X)||^2 / ||X||^2,
    B = 2 * 1[A_4 > 1152.5507888832508]
            + 1[A_8 > 1166.1889834731694].

These four bins, two layers, two thresholds, and strict comparisons are
fixed. Thresholds came only from the cached original bank's float32
post-second-moment reconstruction. The exact formula, file digests,
mtimes, and authenticated weight provenance are in v1 section 2.

Exactly 8,000,000 samples per replica, batch 65,536 including the unchanged
4,608 final batch. A/B PCG64DXSM seeds respectively 7568343082859363083 and
7568343082859363080, corresponding to old XORs 0x4F4E454849474831 and
0x4F4E454849474832. Preserve every original
`rng.standard_normal((count,1024), dtype=np.float32)` call and the original
float32 forward kernels. No sampling is performed by this design review.

Each pihat_rb is the empirical bin count of replica r divided by 8M.
There are no fixed-bank mixture weights, empirical quantile thresholds,
new random streams, new bins, new layers, node/rank/amplitude sweeps, fit,
or sample-count escalation.

The conditional muhat_rb and unbiased vhat_rb are accumulated from the
actual float32 Z using bounded CPU-float64 shifted sums. t_r is the
same-stream empirical true ReLU(Z) mean. The comparable bank is the
**actual original learned post-13 output** in
`rows_current_transport_shadow_v2/row_0035/baseline.npy[13]`, whose digest
is frozen in v1. No teacher-forced oracle state is used as that comparator.

## Primary: literal oracle Gaussianization mean

For each replica r=A,B, calculate the sole prescribed Gaussian-mixture
consumer

    G_r = sum_b pihat_rb * [muhat_rb * Phi(muhat_rb/sqrt(vhat_rb))
                           + sqrt(vhat_rb) * phi(muhat_rb/sqrt(vhat_rb))],
    g_r = G_r - t_r,
    b_r = original_bank_post13 - t_r.

Define the signed cross-replica energies and single-replica noise measures

    E_G = mean_i(g_Ai * g_Bi),
    E_B = mean_i(b_Ai * b_Bi),
    N_G = .5 * mean_i((g_Ai-g_Bi)^2),
    N_B = .5 * mean_i((b_Ai-b_Bi)^2).

Report these four numbers without clipping, the ordinary MSE for every
g_r and b_r, and the two corresponding ordinary MSE ratios. Neurons share
samples and are not independent replication units.

All replay, input-hash, support, positivity, empirical-algebra, and noise
qualifications below must pass. The fixed adequate-SNR requirements are

    E_B > 0,
    E_G >= 0,
    N_B / E_B <= .10,
    N_G / E_B <= .10.

With qualified SNR, the primary information gate is all three conditions:

    E_G / E_B <= .50,
    mean(g_A^2) / mean(b_A^2) <= .65,
    mean(g_B^2) / mean(b_B^2) <= .65.

Failure of one of these resolved information inequalities closes this
**fixed four-bin oracle Gaussianization mean intervention**. Failed
numerical/SNR qualification, including negative cross energies, is
**inconclusive at this fixed acquisition**. It does not authorize more
samples or altered thresholds. A pass supports only further review of a
conditional-state source, full covariance ownership, and complete cost;
it does not authorize a fit, full carrier, deployment, or submission.

The primary test includes whatever cancellation higher cumulants produce
in the literal Gaussianization consumer. It is still a one-cell oracle
information screen, not a final-raw theorem, not proof that full conditional
covariance can be computed cheaply, and not a candidate adjusted score.

G_r is a nonlinear finite-N plug-in estimator and shares its samples with
t_r. Independent-replica products remove random variation about the
plug-in expectation, **not common finite-N plug-in bias** or shared
float32 network arithmetic bias. The population Gaussian-mixture error
and this measured cross energy are not asserted to be exactly equal or
strictly unbiased. No empirical bias correction is fitted.

## Third-moment and replica diagnostics: no veto of a primary pass

Use the same fixed original-bank pre-13 response and third baseline:

    a = -pre13_alpha * pre13_pdf / (6 * maximum(diag(pre13_Cpre),1e-10)),
    k_bank = diag(pre13_K21),
    e_r = a * (khat_total_r-k_bank),
    o_r = a * sum_b pihat_rb * khat_rb.

Report mean(e_A*e_B), mean(o_A*o_B), their signed ratio when its denominator
is positive, ordinary replica squared energies and half squared replica
differences. The former v1 `.50` third ratio is no longer a decision gate.
In particular a large omitted third **cannot veto a direct-mean pass**.
Its purpose is to explain what the conditional representation discards,
not to bound the nonlinear mean consumer or final raw MSE.

Exactly retain the law of total cumulance ownership:

    T = sum_b p_b[T_b + Sym3(mu_b-mu,C_b) + (mu_b-mu)^3].

The discarded third of Gaussianization is sum p_b T_b, not total third
and not bank-minus-total. Empirical algebra uses biased central moments;
unbiased within-bin k-statistics are separate estimates and are not
forced into that finite-sample algebraic equality.

The fixed membership and conditional-variance qualification is:

- At least 8,192 samples per bin in each stream; finite nonnegative
  conditional variances without clipping or repairs.
- For each bin, |p_A-p_B| <= 6*sqrt(2*pbar*(1-pbar)/8,000,000).
- For each bin, relative L2(vhat_A-vhat_B), normalized by the norm of
  their average, <=.02. These are fixed replicate-stability checks,
  not confidence-interval guarantees.
- Save the exact omitted-third replica-difference decomposition

      o_A-o_B = a*sum_b[(p_A-p_B)*(k_A+k_B)/2
                       +(p_A+p_B)/2*(k_A-k_B)],

  including each term's energy and their cross term. Membership and
  conditional-moment estimation are not declared independent.
- Report the count of float32-versus-float64 norm membership changes;
  actual primary memberships always use float64 norm accumulation.

Legacy same-stream replay means and the 32 next-K21 rows must remain
byte-identical; new CPU total third must agree on those 32 diagonals to
relative L2 <=1e-4. Source/target hashes, old teacher-shift limitations,
exact forward/RNG calls, empirical algebra tolerance, and memory limits
remain v1 sections 3--5. A failed fidelity check must stop interpretation,
not invite a new seed or relaxed threshold.

## Required review boundary

Root and the independent agent should review this v2 together with v1's
provenance/acquisition sections and
`certify_cross_depth_energy_total_cumulance_v1.py` before running anything.
The certificate is tiny, target-free, NumPy-only, and currently unexecuted.
No MPS or acquisition code has been launched by this work.

All v1 section-8 representation and cost gaps remain: persistent untied
conditional covariance, norm-conditioning source/PSD, angular versus
ordinary-Gaussian state ownership, component-dependent Gaussian pair maps,
possible factorizations, and nonfree transverse CP trajectories. A later
literal four-component carrier cannot claim its orthogonal three-leg
third tensor is nonzero: the finite-mixture structural limitation holds.
The first/final layer and scheduled component-sharing FLOP estimates
remain illustrative conventional counts, not a metered or deployable card.

Bibliographic correction to v1's link label: the linked Monchot et al.
paper is titled [Input uncertainty propagation through trained neural
networks](https://proceedings.mlr.press/v202/monchot23a.html), PMLR 202,
2023. The PDF URL and Appendix C.1 reference in v1 identify that same
paper; v1's written title was incorrect. This is not a change to the
proposed experiment or an additional literature-derived accuracy claim.
