"""TAG opus-reexam E56 (lead): the pair-carried matrix B in the eigenbasis of the layer's covariance: Rayleigh quotients q_m = u_m' B u_m against lambda_m (what multiple of C
each part of the spectrum carries), how diagonal B is in that basis, and what the fits remove. usage: e56_spectrum.py <npz prefix>"""
import sys, numpy as np
for L in (5, 9, 13):
    z = np.load(sys.argv[1] + '_L%d.npz' % L); B = z['B'].astype(np.float64); C = z['C']; mu = z['mu']; n = len(mu)
    lam, U = np.linalg.eigh(C); lam = lam[::-1]; U = U[:, ::-1]; trC = lam.sum()
    Bt = U.T @ B @ U; q = np.diag(Bt)
    off = ~np.eye(n, dtype=bool)
    a1 = (B[off] * C[off]).sum() / (C[off] ** 2).sum()
    X = np.stack([C[off], (C @ C)[off]], 1); a2, b2 = np.linalg.lstsq(X, B[off], rcond=None)[0]
    v = np.diag(C); Cmu = C @ mu; ad = np.linalg.lstsq(np.stack([trC * v, mu * mu, 2 * mu * Cmu], 1), np.diag(B), rcond=None)[0][0] * trC
    bands = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
    print('layer %d  trC %.3f  lambda1/trC %.4f  a1 %.4f  a(two forms) %.4f  b(two forms) x lambda1 %.4f  a_diag %.4f   mu.u1 cosine %.3f' % (L, trC, lam[0] / trC, a1, a2, b2 * lam[0], ad, abs(mu @ U[:, 0]) / np.linalg.norm(mu)))
    print('   band: mean q/lambda (the multiple of C carried there) | two-form fit a + b lambda | share of |B|^2 in the band diagonal | share in the band block off-diagonal')
    tot = (Bt ** 2).sum()
    for lo, hi in bands:
        blk = Bt[lo:hi, lo:hi]; dg = (np.diag(blk) ** 2).sum(); offb = (blk ** 2).sum() - dg
        print('   %4d-%4d: %.4f | %.4f | %.3f | %.3f' % (lo + 1, hi, (q[lo:hi] / lam[lo:hi]).mean(), a2 + b2 * lam[lo:hi].mean(), dg / tot, offb / tot))
