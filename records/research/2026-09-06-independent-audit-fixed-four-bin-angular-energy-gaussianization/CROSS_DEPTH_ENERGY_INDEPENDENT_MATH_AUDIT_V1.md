# Independent audit of fixed four-bin angular-energy Gaussianization

September6,2026 PDT. Read-only mathematical/source audit by fresh_efficiency.
Reviewed `CROSS_DEPTH_ANGULAR_ENERGY_INFORMATION_PROTOCOL_V2.md`, v1
sections1-5 and8, and the complete unexecuted tiny certificate source.
No samples, target arrays, numerical experiments, extra agent, or MPS.

## Bottom line and one concrete source correction

The law-of-total-cumulance ownership is correct, and changing the primary
to the actual Gaussianization ReLU mean fixes the earlier false necessity
of the K3 projection. The tiny source implements the stated identities
correctly by inspection; this audit is not its execution certificate.

**Before acquisition, fix the comparator's reported-output transform.**
`generate_current_transport_shadow_v2.py` checks
`np.maximum(baseline,0)` against the published parent's means but saves
the raw, unclipped `baseline.npy`. `radial_binary_compact.predict` returns
unclipped stacked means. The current candidate applies maximum0 and casts
float32 at return. Therefore raw `baseline.npy[13]` is not necessarily
the estimator's reported post13 output. Freeze the appropriate maximum0
and documented return-dtype transform in an additive protocol correction,
or verify no affected coordinates and accurately label reference precision.
This issue is independent of the correctly excluded teacher-forced shift.
No target data was needed to discover it.

The current-bank pre13 K21/mean-response source and P0 ownership are
otherwise explicit; the legacy teacher-shift data are used only for
numerical replay. The comparator remains a certified numerical bank
reference, not automatically byte-identical native D4 output merely
because payloads agree.

## 1. Population identities and their limits

For Z_i, bin mean mu_b, variance v_b, third cumulant c_b, unconditional
mu=sum p_b mu_b and d_b=mu_b-mu,

    kappa3(Z_i) = sum_b p_b [c_b+3*d_b*v_b+d_b^3].

A literal mixture of Gaussians with exactly these population p_b,mu_b,v_b
therefore discards exactly sum p_b*c_b. The full tensor formula is the
same with the three permutations of d_b tensor C_b. Between-bin skew is
not missing and cannot be added to the current bank as a fresh source
without subtracting its existing ownership.

The fixed mean-response-weighted K3 norm is NOT a lower bound on ReLU-mean
error. Matching or missing K3 does not determine E[ReLU(Z)]; higher
conditional cumulants can compensate. V2 correctly demotes K3 to a
nonveto diagnostic. The direct mean sum p_b*G(mu_b,v_b) answers the
literal Gaussianization question, where G=mu*Phi(mu/sqrt(v))+sqrt(v)*phi.

Positive homogeneity makes A4/A8 radius-independent for the exact
bias-free Gaussian-input model. Float32 input/forward arithmetic can
leave shared numerical deviations; the protocol appropriately does not
claim its replica comparison measures that shared forward bias.

## 2. Conditional finite-sample estimators and membership

For FIXED bin boundaries, conditional on counts N_b, observations in bin b
are iid from the conditional law. The stated unbiased sample variance and
third k-statistic are correct when N_b>2. Crucially, pihat_b=N_b/N does not
need an independent stream for

    Uhat = sum_b (N_b/N)*khat3_b.

Conditional-on-counts unbiasedness followed by expectation over N_b gives
E[Uhat]=sum p_b*c_b (apart from the explicitly handled unsupported tiny
counts). Reusing the same membership counts is not an extra leading bias.
Membership and within-bin estimation fluctuations still need not be
independent; the exact replica-difference decomposition rightly keeps
their cross term.

The EXACT finite-atom total-cumulance identity must use empirical M2/M3,
as written. Replacing them by corrected k-statistics destroys that exact
sample identity. In particular khat_total minus the third of the mixture
constructed from sample muhat/unbiased vhat need not equal Uhat at finite
N. Keep that numerical identity separate from population unbiasedness.

An arbitrary fixed or data-dependent subtraction shift does not change
exact central moments; its risks here are numerical, not new population
ownership. CPU-float64 arithmetic on the actual float32 Z avoids the
earlier subtract-and-add float32 reconstruction ambiguity.

## 3. Direct-mean plug-in bias

G(muhat_b,vhat_b) is nonlinear, even with unbiased inputs. With M=N_b,
unbiased sample variance, and smooth positive v, its leading conditional
bias is

    [G_mumu*v + 2*G_muv*c3 + G_vv*(c4+2*v^2)]/(2*M) + O(M^-2),

where c4 is the conditional fourth cumulant and

    G_mumu = phi(alpha)/sqrt(v),
    G_muv = -alpha*phi(alpha)/(2*v),
    G_vv = (alpha^2-1)*phi(alpha)/(4*v^(3/2)).

This is a qualification formula, not a supplied rigorous finite-N bound.
The empirical pi weighting cancels the leading count factor in pi/M,
but cannot make this curvature bias zero. The same-stream empirical true
ReLU mean is linear and unbiased; pairing it with G is legitimate and
usually reduces noise. It does not remove G's plug-in bias.

Root accepted a fixed8-by-1M sufficient-statistic block jackknife
DIAGNOSTIC within each unchanged8M stream. This requires no extra RNG
calls, samples, bins, or forward passes: split only CPU sufficient-stat
accumulation at the fixed1M boundaries. For K=8, report

    bias_hat_G = (K-1)*(mean_j G_leave_j-G_full).

Do not apply it as a new primary correction. All leave-block quantities
must recompute their OWN counts, bin means/variances and true ReLU means.
This estimates leading plug-in bias, not an exact bias guarantee. A
nonnegligible diagnostic warrants qualification/inconclusive reporting,
not threshold/sample tuning. Unsupported leave-block cells must be
reported, not silently merged.

## 4. Cross-replica signal/noise interpretation

For independent full streams, g_r=G_r-t_r and b_r=bank-t_r may have highly
correlated terms WITHIN each stream. That is harmless to cross-stream
independence. If E[g_r]=g+bias, then

    E[mean(g_A*g_B)] = mean((g+bias)^2),

not necessarily the population Gaussianization error squared. The bank
cross product estimates its fixed-reference error energy under the
sampled float32 forward law. Ratios of these estimators are not unbiased.
Negative cross energies must remain signed and are correctly classified
inconclusive rather than clipped to zero.

The half squared replica difference estimates ONE stream's noise energy
in expectation. It is not itself a confidence interval. N/E<=.1 is a
predeclared opportunity heuristic;1024 correlated neurons are not1024
independent experimental replications. The two ordinary MSE-ratio checks
also are not held-out prediction scores: each Gaussian mixture is
estimated on its own t_r data. Their role is a noise/stability guard.

The same fixed block statistics can provide an approximate uncertainty
diagnostic for cross energies/ratios: delete each block of A while holding
B fixed, then vice versa, and add the two independent-stream first-order
jackknife variance contributions. Do not call such an8-block approximation
a rigorous confidence guarantee. Borderline crossings need appropriately
cautious interpretation, not a declaration of population separation.

## 5. Valid scope of any eventual verdict

A resolved failure closes the preregistered four fixed A4/A8 bins,
rank35, oracle conditional FIRST/TWO moments, literal Gaussian ReLU
consumer at pre/post13, under the specified practical gates. It does not
close all conditioning, all four-component models, learned conditional
corrections, non-Gaussian transverse carriers, or final-layer scores.
Failed replay, support, bias/noise qualification is inconclusive at this
fixed acquisition, not physical evidence against the representation.

A strong pass still would not provide conditional moment sources, full
conditional covariance propagation, PSD control, affordable splits, or a
working estimator. Rank35 is an observer-fit exploratory cell, not the
independent14-net validation panel. Later source/cost/independent-network
validation remains necessary. All stated no-sweep and no-acquisition
boundaries remain intact.
