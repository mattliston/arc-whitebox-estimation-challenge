"""TAG fable-engineer (E-64b): a loop-form torch drop-in for the torch card's _pair_series_fused (zero.py W-G/W-H broadcast form).
The refit profile (E-64) puts the pair map at ~41-47% of the torch card's forward, memory-bandwidth-bound: the broadcast form
materializes (6,3,n,n) and eight (3,3,n,n) float32 temporaries per call (~700 MB of traffic x 27 calls per net). This computes the same
Hermite series term by term, accumulating (3,n,n) tensors: same products and terms, only the summation order differs (rounding level;
the native comment records the old loop vs the broadcast form at rel rms 8e-8).
usage: import pair_series_torch as ps; ps.install(card, torch)   (replaces env['_pair_series_fused'])"""


def install(card, torch):
    t = torch

    def series(F, G, rho, T21, T22, T31, t3, t4, n):
        # F, G: (3, R, n) float32 (channels, Hermite orders, units); rho, T21, T22, T31: (n, n); t3, t4: (n,)
        P = [rho]
        for k in range(2, 7):
            P.append(P[-1] * (rho / float(k)))     # P[r-1] = rho^r / r!, as the cumulative product of rho / k
        f = lambda r: F[:, r, :, None]
        g = lambda r: G[:, r, None, :]
        ans = P[0] * (f(1) * g(1))
        for r in range(2, 7):
            ans = ans + P[r - 1] * (f(r) * g(r))
        RR = (None, P[0], P[1])                    # weights 1, rho, rho^2 / 2 of the three tri terms

        def tri(fo, go, R, fs=None, gs=None):
            # sum_j R[j] * (F_{fo+j} * G_{go+j}), the broadcast form's tri(A, B, R) with A = Fr[fo:fo+len(R)], B = Gr[go:go+len(R)]
            acc = None
            for j, w in enumerate(R):
                a = f(fo + j) if fs is None else fs * f(fo + j)
                b = g(go + j) if gs is None else gs * g(go + j)
                term = a * b if w is None else w * (a * b)
                acc = term if acc is None else acc + term
            return acc
        e21 = T21[None] * tri(2, 1, RR) + T21.T[None] * tri(1, 2, RR)
        e22 = T22[None] * tri(2, 2, RR)
        e31 = T31[None] * tri(3, 1, RR) + T31.T[None] * tri(1, 3, RR)
        P12 = (P[0], P[1])
        t3c, t4c = t3[None, :, None], t4[None, :, None]
        t3r, t4r = t3[None, None, :], t4[None, None, :]
        e3 = tri(4, 1, P12, fs=t3c) + tri(1, 4, P12, gs=t3r)
        e4 = tri(5, 1, P12, fs=t4c) + tri(1, 5, P12, gs=t4r)
        return ans + e21 / 2 + e22 / 4 + (e31 + e3) / 6 + e4 / 24

    card.env['_pair_series_fused'] = series
    return card
