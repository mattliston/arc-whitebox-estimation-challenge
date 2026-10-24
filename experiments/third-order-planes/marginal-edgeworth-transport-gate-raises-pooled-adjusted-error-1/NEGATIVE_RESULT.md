# Non-Gaussian CP inheritance gate: completed negative

Phase 2 only. The fixed correction passed its algebra and native execution
checks but worsened both development networks. It is closed at the declared
development gate. Weight5 remains the best qualified local candidate.

## Fixed hypothesis and scope

Parent: exact Weight5, SHA256
`a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9`.
Candidate: SHA256
`decef858c58303525f93da024f85a475b6b4f27b87595b19b139904ba7ac13e5`.
The builder preserves the parent as an exact byte prefix and proves the
inverse of four unique replacements in the last `_predict_binary` definition.

With alpha = mu/sigma, gamma3 = k3/sigma^3 and gamma4 = k4/sigma^4:

    gE = Phi(alpha) + phi(alpha) *
         [gamma3*(alpha^2-1)/6 - gamma4*(alpha^3-3*alpha)/24]
    gCP = clip(gE, 0, 1)

This integrates the first-order marginal Edgeworth density over the positive
half-line, followed by the probability projection declared in PROTOCOL.md.
The approximation gates all three inherited CP tensor legs, its repeated
plane, and the shared-old basis consistently. Rebuilding the original star
restores the target repeated plane in real arithmetic. Original FP32 factor
storage and FP64 plane assembly remain; this is not a claim of bit-exact
physical-plane restoration.

Gaussian gates remain in learned features, newborn source, BD/NODE, masks,
trace/radial updates and other consumers. This is an isolated CP inheritance
approximation, not a complete Edgeworth closure. Existing BD gate jets alter
newborn factors; the present change alters historical factors. No fit, new
dataset, rank change, or new persistent large array was introduced.

## Verification and measured result

Independent 192-node Gauss-Legendre integration of the explicit Hermite
density passes all 44 fixed fixtures: maximum absolute error
2.6367796834847468e-14 against the 2e-12 limit. Dense three-leg gating,
inherited-plane, star-restoration and shared-basis tests have maximum error
2.220446049250313e-16. Zero cumulants recover the Gaussian gate exactly.
The synthetic negative-kurtosis stress fixture exceeds raw probability one
on three coordinates, and the declared projection handles those cases.
Actual network clipping frequencies were not instrumented.

Full native 35/100/35 and independent cold100 produce finite, nonnegative
FP32 outputs at all 16 layers. Own repeat, warm/cold100, and the first two
served layers versus Weight5 are byte exact. Outputs, source, weights and
complete bills were sealed before opening the existing 1B reference means.

| Development network | Weight5 raw MSE | Candidate raw MSE | Candidate own-cold FLOPs | Candidate adjusted | Adjusted / Weight5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 35 | 1.871255439571699e-8 | 1.8829235314153358e-8 | 405548334015 | 3.4725258103350366e-9 | 1.0062368833100865 |
| 100 | 1.854682334048165e-8 | 1.871009454023561e-8 | 408576506924 | 3.4763184300878806e-9 | 1.0088046306080374 |

The pooled adjusted ratio is **1.007519821658286**, a 0.751982% worsening.
Raw ratios are 1.0062354350971492 and 1.008803189460353. Both individual
improvement requirements and the pooled <=0.99 requirement fail. These are
two reused development networks with ordinary 1B errors, not the 14-network
noise-corrected panel or independent validation.

## Resources and disposition

Four complete prediction calls total 1,594,369,321,838 billed FLOPs. Each costs
exactly 583,680 more FLOPs than its corresponding Weight5 call. Maximum
externally monitored group RSS is 6,422,659,072 bytes; the maximum native
self-report is 6,423,199,744 bytes. Added swap is zero. Every owned guarded
group exited successfully and retired. Short unit/scorer polling samples
are not interpreted as their actual memory peaks.

No strength, order, layer-window, mask, rank, precision or fit sweep follows.
No wider panel, client qualification, package or upload follows this failed
development gate. The saved sources, outputs, bills and watches are covered
by closeout.json. The reviewed earlier source routes remain documented in
source_review.json; this result does not rule out all non-Gaussian transport.

Weight5 remains locally qualified at 14-network noise-corrected adjusted
3.883207214854186e-9. Its exact archive approval is pending and no upload
receipt exists. Official best remains graded submission #331584 at adjusted
4.015535762022727e-9. The approximately 1e-9 goal remains active and unmet.
