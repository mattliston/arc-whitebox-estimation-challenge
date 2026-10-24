"""TAG opus-physicist SO2: the SECOND-ORDER terms in the third cumulants of the card's scalar map and pair map (certificate:
t15_so2_cert.py, PASS: with them the error of every moment falls as eps^3 instead of eps^2 on an exactly solvable non-Gaussian pair).
The card's maps hold the Gaussian part and the terms linear in (k3, K21) and in (k4, K22, K31). At the order of the fourth cumulants
the expansion also has the products of two third cumulants. With t3 = k3 / sigma^3, T[i, j] = K21[i, j] / (v_i sigma_j) (zero diagonal),
rho = C_ij / (sigma_i sigma_j), A_r = sigma^(r-1) E_G relu^(r) (the card's `derivatives`), P_k = rho^k / k!:
  single unit:  mean += t3^2 sigma A_6 / 72;  E relu^2 += t3^2 v (2 A_5) / 72;  E relu^3 += t3^2 sigma^3 A_4 / 12;  E relu^4 += t3^2 v^2 A_3 / 3
  pair, covariance / (sigma_i sigma_j) += (1/72) [ a + a' + b + b' + e + e' + c + d ],   ' = transpose, and
     a = 6 t3_i T_ij            sum_{k>=0} P_k A_(5+k)(i) A_(1+k)(j)
     b = 6 t3_i T_ji            sum_{k>=0} P_k A_(4+k)(i) A_(2+k)(j)
     q = 9 T_ij^2               sum_{k>=0} P_k A_(4+k)(i) A_(2+k)(j)
     c = 18 T_ij T_ji           sum_{k>=0} P_k A_(3+k)(i) A_(3+k)(j)
     d = 2 t3_i t3_j            sum_{k>=1} P_k A_(3+k)(i) A_(3+k)(j)
     e = t3_i^2                 sum_{k>=1} P_k A_(6+k)(i) A_k(j)
  (the k = 0 parts of d and e are products of single-unit means and are not part of the covariance).
  READING: a + b' are the card's K21 term (1/2) T_ij A_2(i) A_1(j) with the skew-corrected kink density and gate,
  A_2 -> A_2 + (t3/6) A_5 = phi (1 - (t3/6) He3(alpha)) and A_1 -> A_1 + (t3/6) A_4; d + e are the Gaussian term rho A_1 A_1 with the
  skew-corrected gates (second order); q and c are the genuine products of two K21 entries.
  third-cumulant target: the same list with B (relu^2) on the i side gives E f_i^2 f_j; cum(f_i, f_i, f_j) = that - 2 m_i cov_ij.
install(cd, torch, mean=, raw2=, cov=, tgt=, kmax=, layers=, parts=, final=):
  mean  strength of the mean term at layers lo..min(hi, 14), INSIDE the pair map (host; the per-unit variance = E relu^2 - mean^2 follows)
  raw2  strength of the E relu^2 term (None = the same as mean, the consistent per-unit variance)
  cov   strength of the off-diagonal covariance list;  parts = which of a b q c d e;  kmax = highest power of rho kept (0, 1 or 2)
  tgt   strength of the second-order part of the third-cumulant target (post_k3 / post_k4 on the diagonal, the plane cum(f_i, f_i, f_j) off it),
        always formed from the FULL second-order mean and covariance (a target with half of its second-order terms is worse than none)
  final strength of the mean term at layer 15, added AFTER the learned correction (the learned head keeps its inputs)
Everything is elementwise on arrays the pair map already holds (no matrix product). cd.so2_info[layer] = sizes for the log."""
import math


def _A(t, a, ph, g, mgs, order):
    A = [mgs, g]; h0 = t.ones_like(a); h1 = a
    for r in range(2, order + 1):
        if r == 2: h = h0
        elif r == 3: h = h1
        else: h = a * h1 - (r - 3) * h0; h0, h1 = h1, h
        A.append((-1) ** (r - 2) * h * ph)
    return A


def single(t, mu, v, k3, order=1):
    """second-order additions (mean, E relu^2, E relu^3, E relu^4) for one unit."""
    v = v.clamp_min(1e-10); s = v.sqrt(); a = mu / s; ph = t.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    t3 = k3 / (v * s); q = t3 * t3
    he2, he3, he4 = a * a - 1, a ** 3 - 3 * a, a ** 4 - 6 * a * a + 3
    return q * s * he4 * ph / 72, -q * v * he3 * ph / 36, q * v * s * he2 * ph / 12, -q * v * v * a * ph / 3


def pair_list(t, A, Bi, rho, T, t3, kmax, parts='abqcde'):
    """(1/72) x the connected second-order list; A = list of standardized derivative vectors for the j side, Bi = the same for the i
    side (A itself for the covariance, the relu^2 list for E f_i^2 f_j, where the list is NOT symmetric and both orientations are written out)."""
    P = [None, rho, rho * rho / 2][:kmax + 1]; Tt = T.T
    def D(p, q, k0=0, left=Bi, right=A):
        acc = None
        for k in range(k0, kmax + 1):
            term = t.outer(left[p + k], right[q + k]); term = term if P[k] is None else P[k] * term
            acc = term if acc is None else acc + term
        return acc
    sym = Bi is A; X = t.zeros_like(rho); ti, tj = t3[:, None], t3[None, :]
    if 'a' in parts:
        Y = 6 * ti * T * D(5, 1); X = X + Y + (Y.T if sym else 6 * tj * Tt * D(1, 5))
    if 'b' in parts or 'q' in parts:
        cb, cq = 6.0 * ('b' in parts), 9.0 * ('q' in parts)
        Y = (cb * ti * Tt + cq * T * T) * D(4, 2); X = X + Y + (Y.T if sym else (cb * tj * T + cq * Tt * Tt) * D(2, 4))
    if 'c' in parts: X = X + 18 * T * Tt * D(3, 3)
    if 'd' in parts and kmax >= 1: X = X + 2 * ti * tj * D(3, 3, 1)
    if 'e' in parts and kmax >= 1:
        Y = ti * ti * D(6, 0, 1); X = X + Y + (Y.T if sym else tj * tj * D(0, 6, 1))
    return X / 72


def install(cd, torch, mean=0.0, raw2=None, cov=0.0, tgt=0.0, kmax=1, layers=(1, 15), parts='abqcde', final=0.0):
    t = torch; f64 = t.float64; lo, hi = layers; cd.so2_info = {}; raw2 = mean if raw2 is None else raw2
    pair0 = cd.env['_nonlinear_pair_batch']
    offd = lambda x: x - t.diag(t.diagonal(x))

    def pair(mu, C, K21, k4, k22_override=None, metric='identity', **kw):
        m, covm, post21, radial, aux = pair0(mu, C, K21, k4, k22_override, metric, **kw)
        l = cd.call_layer
        if not (lo <= l <= min(hi, 14)) or not (mean or raw2 or cov or tgt): return m, covm, post21, radial, aux
        s, a, ph, g, mg = (aux[k].to(f64) for k in ('sigma', 'alpha', 'pdf', 'gate', 'gaussian_mean')); v = s * s
        k3 = t.diagonal(K21).to(f64); t3 = k3 / (v * s); k4v = (k4 if t.is_tensor(k4) else t.tensor(float(k4), dtype=f64)).to(f64).expand(v.shape)
        m2, q2, q3, q4 = single(t, mu.to(f64), v, k3)
        d, e = ph / s, -a * ph / v
        r2g = (mu * mu + v) * g + mu * s * ph; r3g = mu * (mu * mu + 3 * v) * g + (mu * mu + 2 * v) * s * ph
        r4g = (mu ** 4 + 6 * mu * mu * v + 3 * v * v) * g + mu * (mu * mu + 5 * v) * s * ph
        r2 = r2g + k3 * d / 3 + k4v * e / 12; r3 = r3g + k3 * g + k4v * d / 4; r4 = r4g + 4 * k3 * mg + k4v * g      # the card's own raw moments (zero.py)
        mn = m + mean * m2; var = (r2 + raw2 * q2) - mn * mn
        info = dict(mean_rel=float((mean * m2).norm() / m.norm()), var_rel=float((var - t.diagonal(covm)).norm() / t.diagonal(covm).norm()),
                    t3_rms=float((t3 * t3).mean().sqrt()), var_parity=float((r2 - m * m - t.diagonal(covm)).abs().max() / t.diagonal(covm).abs().max()))
        covn, p21n = covm, post21; aux = dict(aux)
        if cov or tgt:
            rho = offd(C / t.outer(s, s)); T = offd(K21 / (v[:, None] * s[None, :])); off = offd(covm)
            A = _A(t, a, ph, g, mg / s, kmax + 7)
            dc1 = offd(pair_list(t, A, A, rho, T, t3, kmax, parts) * t.outer(s, s))
            info.update(cov_rel=float(dc1.norm() / off.norm()), T_rms=float((T * T).mean().sqrt()), rho_rms=float((rho * rho).mean().sqrt()))
            if cov: covn = covm + cov * dc1
            if tgt:
                B = [(a * a + 1) * g + a * ph] + [2 * x for x in A[:-1]]
                dp1 = pair_list(t, A, B, rho, T, t3, kmax, parts) * t.outer(v, s)
                m1 = m + m2; dt = offd(dp1 - 2 * (m1[:, None] * (off + dc1) - m[:, None] * off))
                r2f = r2 + q2; pk3 = (r3 + q3) - 3 * m1 * r2f + 2 * m1 ** 3; pk3_0 = r3 - 3 * m * r2 + 2 * m ** 3
                info.update(tgt_rel=float(dt.norm() / offd(post21).norm()), k3post_rel=float((pk3 - pk3_0).norm() / pk3_0.norm()),
                            k3post_parity=float((pk3_0 - t.diagonal(post21)).abs().max() / t.diagonal(post21).abs().max()))
                p21n = post21 + tgt * t.diagonal_scatter(dt, pk3 - pk3_0)
                aux['post_k3'] = t.diagonal(p21n)
        if mean or raw2:
            covn = t.diagonal_scatter(covn, var); aux['post_v'] = var
        cd.so2_info[l] = info
        return mn, covn, p21n, radial, aux
    cd.env['_nonlinear_pair_batch'] = pair
    if final and hi >= 15:
        old = cd.hooks.get('delta')
        def delta(l, dl, ctx):
            if old is not None: dl = old(l, dl, ctx)
            if l != 15: return dl
            m2 = single(t, ctx['mu'].to(f64), ctx['vv'].to(f64), ctx['aux']['pre_k3'].to(f64))[0]
            cd.so2_info[15] = dict(mean_rel=float((final * m2).norm() / ctx['host'].norm()))
            return dl + (final * m2).to(dl.dtype)
        cd.hooks['delta'] = delta
    return cd
