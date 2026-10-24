"""HELD fixed moment-matched cubic-Gaussian ReLU mean; no model authority.

Z is standard normal and Y=a*H1(Z)+b*H2(Z)+c*H3(Z).
Unit variance imposes a=sqrt(1-2*b*b-6*c*c). The two polynomial
cumulant equations below are exact Gaussian integrals, not fitted weights.
Only a converged strictly increasing cubic is used. Every other unit falls
back to its original scalar mean. Solver bounds/tolerances are numerical
safeguards; no parameter is selected from predictions or truth.
"""


def cubic_marginal_mean(mu, sigma, k3, k4, baseline_mean, fnp, normal_cdf):
    s = sigma.astype(fnp.float64)
    alpha = mu / s
    skew = k3 / (s*s*s)
    kurt = k4 / (s*s*s*s)
    b = fnp.clip(skew/6.0, -0.25, 0.25)
    c = fnp.clip((kurt-(4.0/3.0)*skew*skew)/24.0, -0.25, 0.25)
    valid = fnp.isfinite(skew) & fnp.isfinite(kurt) & fnp.isfinite(alpha)

    # Eight fixed Newton steps, each limited to a numerical trust region.
    for _ in range(8):
        a = fnp.sqrt(1.0-2.0*b*b-6.0*c*c)
        a2, b2, c2 = a*a, b*b, c*c
        S = 6*a2*b + 36*a*b*c + 8*b2*b + 108*b*c2
        K = (24*a2*a*c + 48*a2*b2 + 216*a2*c2 + 576*a*b2*c
             + 1296*a*c2*c + 48*b2*b2 + 2160*b2*c2 + 3240*c2*c2)
        Sa = 12*a*b + 36*b*c
        Sb = 6*a2 + 36*a*c + 24*b2 + 108*c2
        Sc = 36*a*b + 216*b*c
        Ka = 72*a2*c + 96*a*b2 + 432*a*c2 + 576*b2*c + 1296*c2*c
        Kb = 96*a2*b + 1152*a*b*c + 192*b2*b + 4320*b*c2
        Kc = 24*a2*a + 432*a2*c + 576*a*b2 + 3888*a*c2 + 4320*b2*c + 12960*c2*c
        ab, ac = -2*b/a, -6*c/a
        j11, j12 = Sb+Sa*ab, Sc+Sa*ac
        j21, j22 = Kb+Ka*ab, Kc+Ka*ac
        det = j11*j22-j12*j21
        good = fnp.isfinite(det) & (fnp.abs(det)>1e-12)
        valid = valid & good
        safe_det = fnp.where(good, det, 1.0)
        db = ((S-skew)*j22-(K-kurt)*j12)/safe_det
        dc = (j11*(K-kurt)-j21*(S-skew))/safe_det
        step = fnp.minimum(1.0, 0.05/fnp.maximum(fnp.maximum(fnp.abs(db),fnp.abs(dc)),1e-30))
        b = fnp.clip(b-step*db, -0.25, 0.25)
        c = fnp.clip(c-step*dc, -0.25, 0.25)

    a = fnp.sqrt(1.0-2*b*b-6*c*c)
    a2, b2, c2 = a*a, b*b, c*c
    S = 6*a2*b + 36*a*b*c + 8*b2*b + 108*b*c2
    K = (24*a2*a*c + 48*a2*b2 + 216*a2*c2 + 576*a*b2*c
         + 1296*a*c2*c + 48*b2*b2 + 2160*b2*c2 + 3240*c2*c2)
    # Derivative minimum is a-3c-b^2/(3c); Gaussian limit handled exactly.
    gaussian = (skew==0.0) & (kurt==0.0)
    monotone = (c>0.0) & ((a-3*c)-b2/(3*fnp.maximum(c,1e-30))>0.0)
    valid = (valid & monotone & (fnp.abs(S-skew)<=1e-8)
             & (fnp.abs(K-kurt)<=1e-8)) | gaussian
    # Invalid units use the Gaussian polynomial during the root calculation,
    # then receive their untouched baseline mean below.
    aa = fnp.where(valid,a,1.0)
    bb = fnp.where(valid,b,0.0)
    cc = fnp.where(valid,c,0.0)
    lo = fnp.full_like(alpha,-40.0)
    hi = fnp.full_like(alpha,40.0)
    flo = alpha+aa*lo+bb*(lo*lo-1)+cc*(lo*lo*lo-3*lo)
    fhi = alpha+aa*hi+bb*(hi*hi-1)+cc*(hi*hi*hi-3*hi)
    valid = valid & (flo<=0.0) & (fhi>=0.0)
    for _ in range(48):
        mid = (lo+hi)*0.5
        value = alpha+aa*mid+bb*(mid*mid-1)+cc*(mid*mid*mid-3*mid)
        lo = fnp.where(value<=0.0,mid,lo)
        hi = fnp.where(value>0.0,mid,hi)
    z = (lo+hi)*0.5
    phi = fnp.exp(-0.5*z*z)*0.39894228040143267794
    proposal = s*(alpha*normal_cdf(-z)+(aa+bb*z+cc*(z*z-1))*phi)
    valid = valid & fnp.isfinite(proposal) & (proposal>=0.0)
    return fnp.where(valid,proposal,baseline_mean),valid
