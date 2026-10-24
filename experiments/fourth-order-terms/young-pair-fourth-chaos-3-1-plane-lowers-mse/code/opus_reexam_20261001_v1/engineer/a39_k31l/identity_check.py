"""TAG opus_engineer a39 K31L4, the algebra the native form rests on (plain numpy, seconds): the lead's fourth-chaos (3,1) plane of one
cohort (O/k31l.py, chaos4=True; spec O/NCAR6_SPEC.md addendum),
    K31L4_ij = sum_h [AY_ih Y_jh + AE_ih E_jh],  Y = p - t+ E, E = (p - q) / gap,
written on the pair's two pushed columns p (+), q (-) and D = p - q with K4L's own four birth coefficients e1..e4
(Q = e1 p^3 D + e2 p^2 D^2 + e3 p D^3 + e4 D^4 is the hub's local quartic) and the three column means of p^2, p D, D^2:
    cubic part      (1/4) [dQ/dp_i p_j + dQ/dD_i D_j]
    its projection  (1/8) [dH/dp_i p_j + dH/dD_i D_j],  H = m_pp Q_pp + 2 m_pD Q_pD + m_DD Q_DD  (a quadratic form in (p, D))
so the left factor of the read gets, for the + column  (1/4)(Q_p + Q_D) - (1/8)(H_p + H_D)  and for the - column  -(1/4) Q_D + (1/8) H_D.
usage: identity_check.py"""
import numpy as np
rng = np.random.default_rng(3)
n, K = 240, 37
p = rng.standard_normal((n, K)) * 0.3; q = p + rng.standard_normal((n, K)) * 0.2
tp = rng.uniform(0.4, 2.0, K); tm = -rng.uniform(0.4, 2.0, K); gap = tp - tm
S1, S2, S3, S4 = (rng.standard_normal(K) for _ in range(4))
# ---- the reference (k31l.py, chaos4)
E = (p - q) / gap; Y = (tp * q - tm * p) / gap
AY = 3 * S1 * Y * Y * E + 3 * S2 * Y * E * E + S3 * E ** 3
AE = S1 * Y ** 3 + 3 * S2 * Y * Y * E + 3 * S3 * Y * E * E + S4 * E ** 3
plane_full = AY @ Y.T + AE @ E.T
mY2, mYE, mE2 = (Y * Y).mean(0), (Y * E).mean(0), (E * E).mean(0)
AY4 = AY - 3 * (S1 * (mY2 * E + 2 * mYE * Y) + S2 * (mE2 * Y + 2 * mYE * E) + S3 * mE2 * E)
AE4 = AE - 3 * (S1 * mY2 * Y + S2 * (mY2 * E + 2 * mYE * Y) + S3 * (mE2 * Y + 2 * mYE * E) + S4 * mE2 * E)
plane4 = AY4 @ Y.T + AE4 @ E.T
# ---- K4L's coefficients: Q = 4 S1 Y^3 E + 6 S2 Y^2 E^2 + 4 S3 Y E^3 + S4 E^4 with Y = p - (t+ / gap) D, E = D / gap, as e1 p^3 D + e2 p^2 D^2 + e3 p D^3 + e4 D^4
tau = tp / gap; ig = 1.0 / gap
e1 = 4 * S1 * ig
e2 = -12 * S1 * tau * ig + 6 * S2 * ig ** 2
e3 = 12 * S1 * tau ** 2 * ig - 12 * S2 * tau * ig ** 2 + 4 * S3 * ig ** 3
e4 = -4 * S1 * tau ** 3 * ig + 6 * S2 * tau ** 2 * ig ** 2 - 4 * S3 * tau * ig ** 3 + S4 * ig ** 4
D = p - q
Q = 4 * S1 * Y ** 3 * E + 6 * S2 * Y * Y * E * E + 4 * S3 * Y * E ** 3 + S4 * E ** 4
print('quartic in (p, D):', np.abs(Q - (e1 * p ** 3 * D + e2 * p * p * D * D + e3 * p * D ** 3 + e4 * D ** 4)).max() / np.abs(Q).max())
Qp = 3 * e1 * p * p * D + 2 * e2 * p * D * D + e3 * D ** 3
QD = e1 * p ** 3 + 2 * e2 * p * p * D + 3 * e3 * p * D * D + 4 * e4 * D ** 3
mine_full = 0.25 * (Qp @ p.T + QD @ D.T)
print('cubic part:', np.abs(mine_full - plane_full).max() / np.abs(plane_full).max())
mpp, mpD, mDD = (p * p).mean(0), (p * D).mean(0), (D * D).mean(0)
Hp = p * (12 * e1 * mpD + 4 * e2 * mDD) + D * (6 * e1 * mpp + 8 * e2 * mpD + 6 * e3 * mDD)
HD = p * (6 * e1 * mpp + 8 * e2 * mpD + 6 * e3 * mDD) + D * (4 * e2 * mpp + 12 * e3 * mpD + 24 * e4 * mDD)
mine4 = mine_full - 0.125 * (Hp @ p.T + HD @ D.T)
print('fourth-chaos plane:', np.abs(mine4 - plane4).max() / np.abs(plane4).max())
# ---- as left-factor additions on the two columns
cP = 0.25 * (Qp + QD) - 0.125 * (Hp + HD); cM = -0.25 * QD + 0.125 * HD
print('as columns:', np.abs((cP @ p.T + cM @ q.T) - plane4).max() / np.abs(plane4).max())
