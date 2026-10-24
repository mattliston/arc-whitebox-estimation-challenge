"""TAG opus_engineer a36 (light, under a second, no estimator): what flopscope bills for the pieces a sketched carrier needs, read from
its own budget counter on small real calls. usage: price_probe.py"""
import numpy as np
import flopscope as flops
fnp = flops.numpy
n = 1024
rng = np.random.default_rng(3)


def bill(f, *arrays):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ops = [fnp.asarray(a) for a in arrays]
        before = int(ctx.summary_dict()['flops_used'])
        out = f(*ops)
        return int(ctx.summary_dict()['flops_used']) - before, len(ctx._op_log), out


W = rng.standard_normal((n, n)).astype(np.float32); C = (W @ W.T / n).astype(np.float32)
for k in (8, 16, 24, 32, 56):
    Y = rng.standard_normal((n, k)).astype(np.float32)
    print('k = %2d: C @ Y %d flops (%.3f n^2 k); W.T @ Y %d; QR of n x k (reduced) %d flops (%.2f Mflop); Y.T @ Y %d; Y @ M (k x k) %d' % (
        k, bill(lambda c, y: c @ y, C, Y)[0], bill(lambda c, y: c @ y, C, Y)[0] / n / n / k, bill(lambda w, y: w.T @ y, W, Y)[0], bill(lambda y: fnp.linalg.qr(y), Y)[0], bill(lambda y: fnp.linalg.qr(y), Y)[0] / 1e6,
        bill(lambda y: y.T @ y, Y)[0], bill(lambda y, m: y @ m, Y, rng.standard_normal((k, k)).astype(np.float32))[0]))
for k in (8, 16, 32, 56):
    S = rng.standard_normal((k, k)); S = (S + S.T).astype(np.float32)
    print('symmetric eigen-decomposition of a %2d x %2d matrix (eigh): %d flops (float32 input); float64 input: %d' % (k, k, bill(lambda s: fnp.linalg.eigh(s), S)[0], bill(lambda s: fnp.linalg.eigh(s), S.astype(np.float64))[0]))
Y = rng.standard_normal((n, 24)).astype(np.float32)
print('QR mode r of n x 24: %d; svd of n x 24 (full_matrices=False): %d; eigh(1024): %d (%.2f Gflop)' % (bill(lambda y: fnp.linalg.qr(y, mode='r'), Y)[0], bill(lambda y: fnp.linalg.svd(y, full_matrices=False), Y)[0], flops.numpy.linalg._decompositions.eigh_cost(n) if hasattr(flops.numpy.linalg, '_decompositions') else 9 * n ** 3, 9 * n ** 3 / 1e9))
print('probes from weight columns: W[:, :24] as an operand: %d flops (a view); W[:, idx] with an index array: %d flops' % (bill(lambda w: w[:, :24], W)[0], bill(lambda w: w[:, fnp.asarray(np.arange(0, 1024, 43))], W)[0]))
print('rank-32 plane Z M Z^T: (Z @ M) %d + (ZM) @ Z.T %d flops; standardise rows and columns (einsum ij,i,j->ij) %d; add into a plane %d; transpose-add %d' % (
    bill(lambda z, m: z @ m, rng.standard_normal((n, 32)).astype(np.float32), rng.standard_normal((32, 32)).astype(np.float32))[0],
    bill(lambda a, b: a @ b.T, rng.standard_normal((n, 32)).astype(np.float32), rng.standard_normal((n, 32)).astype(np.float32))[0],
    bill(lambda p, a, b: fnp.einsum('ij,i,j->ij', p, a, b, optimize=False), C, rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32))[0],
    bill(lambda p, q: p + q, C, C.copy())[0], bill(lambda p: p + p.T, C)[0]))
g = rng.uniform(0, 1, n).astype(np.float32); Z = rng.standard_normal((n, 32)).astype(np.float32)
print('gate a block (g[:, None] * Z, 32 columns): %d flops; float64 block products cost %.1fx (C64 @ Y64: %d)' % (bill(lambda a, z: a[:, None] * z, g, Z)[0],
      bill(lambda c, y: c @ y, C.astype(np.float64), Z.astype(np.float64))[0] / bill(lambda c, y: c @ y, C, Z)[0], bill(lambda c, y: c @ y, C.astype(np.float64), Z.astype(np.float64))[0]))
print('random numbers: fnp.random available: %s' % hasattr(fnp, 'random'))
try:
    print('  standard_normal((1024, 16)) bills %d flops' % bill(lambda: fnp.random.default_rng(1).standard_normal((n, 16)))[0])
except Exception as e:
    print('  not usable:', type(e).__name__, str(e)[:120])
