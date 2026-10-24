"""TAG opus_engineer a18 (light, numpy only, a fraction of a second): the per-pair least-squares step of the physicist's F4b
(O/physicist/f4b.py: Gram of the three matrix types, 3 x 2 design, batched 2 x 2 solve) against a closed form that needs only the
Gram of the pair's two gated columns:
    x_s(k) = sum_s' (G2^-1)_{s s'} (d_k / 2) Q_s'(Y_k, Wt_k),   G2 = [[(u+.u+)^2, (u+.u-)^2], [(u+.u-)^2, (u-.u-)^2]],
    Q_s(y, w) = (S0 A^2 + 2 S1 A B + S2 B^2) y^2 + 2 (S1 A^2 + 2 S2 A B + S3 B^2) y w + (S2 A^2 + 2 S3 A B + S4 B^2) w^2,
    A = a.u_s, B = b.u_s  (a = gated Y, b = gated Wt, u_s = a + t_s b the pair's gated columns).
usage: check_projection.py"""
import numpy as np
rng = np.random.default_rng(3)
n, K = 300, 7
a = rng.standard_normal((n, K)); b = rng.standard_normal((n, K)) * 0.7 + 0.3 * a
tp = rng.uniform(0.4, 2.0, K); tm = -rng.uniform(0.4, 2.0, K)
S = [np.zeros(K)] + [rng.standard_normal(K) for _ in range(4)]
# the physicist's route
aa, bb, ab = (a * a).sum(0), (b * b).sum(0), (a * b).sum(0)
Gam = np.stack((np.stack((aa * aa, 2 * aa * ab, ab * ab)), np.stack((2 * aa * ab, 2 * (aa * bb + ab * ab), 2 * bb * ab)), np.stack((ab * ab, 2 * bb * ab, bb * bb)))).transpose(2, 0, 1)
one = np.ones(K)
Bm = np.stack((np.stack((one, one)), np.stack((tp, tm)), np.stack((tp ** 2, tm ** 2)))).transpose(2, 0, 1)
BtG = Bm.transpose(0, 2, 1) @ Gam; G2 = BtG @ Bm
R = np.linalg.solve(G2, BtG)
M = np.stack((np.stack((S[0], 2 * S[1], S[2])), np.stack((S[1], 2 * S[2], S[3])), np.stack((S[2], 2 * S[3], S[4])))).transpose(2, 0, 1)
Nm = R @ M                                                    # K x 2 x 3 on (Y^2, Y Wt, Wt^2)
# closed form from the Gram of the two gated columns
up, um = a + tp * b, a + tm * b
spp, spq, sqq = (up * up).sum(0), (up * um).sum(0), (um * um).sum(0)
G = tp - tm
A = np.stack(((tp * spq - tm * spp) / G, (tp * sqq - tm * spq) / G)); B = np.stack(((spp - spq) / G, (spq - sqq) / G))     # (2, K): a.u_s, b.u_s
assert np.allclose(A[0], (a * up).sum(0)) and np.allclose(B[1], (b * um).sum(0))
c = np.stack((S[0] * A * A + 2 * S[1] * A * B + S[2] * B * B, 2 * (S[1] * A * A + 2 * S[2] * A * B + S[3] * B * B), S[2] * A * A + 2 * S[3] * A * B + S[4] * B * B), axis=1)   # (2, 3, K)
det = spp ** 2 * sqq ** 2 - spq ** 4
mine = np.stack(((sqq ** 2 * c[0] - spq ** 2 * c[1]) / det, (spp ** 2 * c[1] - spq ** 2 * c[0]) / det))                    # (2, 3, K)
print('closed form against the batched solve: max relative difference', float(np.abs(mine.transpose(2, 0, 1) - Nm).max() / np.abs(Nm).max()))
# the same coefficients on (P^2, P Q, Q^2), P = Y + t+ Wt, Q = Y + t- Wt
alpha = (mine[:, 0] * tm ** 2 - mine[:, 1] * tm + mine[:, 2]) / G ** 2
beta = (-2 * mine[:, 0] * tp * tm + mine[:, 1] * (tp + tm) - 2 * mine[:, 2]) / G ** 2
gamma = (mine[:, 0] * tp ** 2 - mine[:, 1] * tp + mine[:, 2]) / G ** 2
Y = rng.standard_normal((n, K)); Wt = rng.standard_normal((n, K)); P, Q = Y + tp * Wt, Y + tm * Wt
for s in (0, 1):
    want = Nm[:, s, 0] * Y * Y + Nm[:, s, 1] * Y * Wt + Nm[:, s, 2] * Wt * Wt
    got = alpha[s] * P * P + beta[s] * P * Q + gamma[s] * Q * Q
    print(f'column {"+-"[s]}: quadratic on (P, Q) against the quadratic on (Y, Wt): max relative difference', float(np.abs(got - want).max() / np.abs(want).max()))

# ---- the pair-basis form: the hub's quartic re-expressed on the pair's own two columns, P = Y + t+ Wt and Q = Y + t- Wt, with
# five coefficients Psi_0..Psi_4 fixed at birth; then the two shifts are
#   x+ = (d/2) [ (Psi0 + k+ Psi1) P^2 + 2 (Psi1 + k+ Psi2) P Q + (Psi2 + k+ Psi3) Q^2 ],   k+ = 2 spq sqq / (spp sqq + spq^2)
#   x- = (d/2) [ (Psi2 + k- Psi1) P^2 + 2 (Psi3 + k- Psi2) P Q + (Psi4 + k- Psi3) Q^2 ],   k- = 2 spq spp / (spp sqq + spq^2)
from math import comb
Psi = np.zeros((5, K))
for m in range(5):
    for i in range(4 - m + 1):
        for j in range(m + 1):
            Psi[m] += comb(4 - m, i) * comb(m, j) * (-tm) ** (4 - m - i) * tp ** (m - j) * (-1) ** j * S[i + j]
Psi /= G ** 4
kp, km = 2 * spq * sqq / (spp * sqq + spq ** 2), 2 * spq * spp / (spp * sqq + spq ** 2)
xp = (Psi[0] + kp * Psi[1]) * P * P + 2 * (Psi[1] + kp * Psi[2]) * P * Q + (Psi[2] + kp * Psi[3]) * Q * Q
xm = (Psi[2] + km * Psi[1]) * P * P + 2 * (Psi[3] + km * Psi[2]) * P * Q + (Psi[4] + km * Psi[3]) * Q * Q
for s, got in ((0, xp), (1, xm)):
    want = Nm[:, s, 0] * Y * Y + Nm[:, s, 1] * Y * Wt + Nm[:, s, 2] * Wt * Wt
    print(f'pair-basis form, column {"+-"[s]}: max relative difference', float(np.abs(got - want).max() / np.abs(want).max()))
# K4L's local term in the same basis (S0 = 0 there): sum_m binom(4, m) Psi_m P^(4-m) Q^m = sum_m binom(4, m) S_m Y^(4-m) Wt^m
lhs = sum(comb(4, m) * Psi[m] * P ** (4 - m) * Q ** m for m in range(5)); rhs = sum(comb(4, m) * S[m] * Y ** (4 - m) * Wt ** m for m in range(5))
print('the quartic itself in the pair basis: max relative difference', float(np.abs(lhs - rhs).max() / np.abs(rhs).max()))
# rule A (f4.py) is the case Psi = (A lam+, 0, 0, 0, A lam-)
Aa = rng.standard_normal(K); lp, lm = rng.uniform(0.5, 2, K), -rng.uniform(0.5, 2, K)
SA = [Aa * (lp * tp ** m + lm * tm ** m) for m in range(5)]
PsiA = np.zeros((5, K))
for m in range(5):
    for i in range(4 - m + 1):
        for j in range(m + 1):
            PsiA[m] += comb(4 - m, i) * comb(m, j) * (-tm) ** (4 - m - i) * tp ** (m - j) * (-1) ** j * SA[i + j]
PsiA /= G ** 4
print('rule A in the pair basis: Psi0 - A lam+', float(np.abs(PsiA[0] - Aa * lp).max()), ' Psi4 - A lam-', float(np.abs(PsiA[4] - Aa * lm).max()), ' |Psi1..3|', float(np.abs(PsiA[1:4]).max()))
