"""[opus-lead] how noisy are the estimator's fast-matmul kernels? Run _shared_old384_mm in-process (after setup) on the dumped L7 join's
start-sketch operands and compare with a float64 reference; also plain fnp.matmul. Reports max-normalised and Frobenius relative errors.
usage (OpenBLAS env python): probe_kernel_noise_v1.py <candidate.py> <join_L7.npz>"""
import sys, types, numpy as np
from pathlib import Path
import flopscope as flops
cand = Path(sys.argv[1]); d = np.load(sys.argv[2])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_pk'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
J = d['J'].astype(np.float32); lam = d['lam'].astype(np.float32); w = lam * lam; r = 384
dg = (J * J) @ w; rows = np.sort(np.argsort(-dg.astype(np.float64), kind='stable')[:r])
cn = ((J * J).sum(0) * w).astype(np.float64); order = np.argsort(-cn, kind='stable'); c0 = np.sort(order[:1024]); c1 = np.sort(order[:2048])
J0 = J[:, c0]; J1 = J[:, c1]
def check(name, A, B):
    ref = A.astype(np.float64) @ B.astype(np.float64)
    fnp = mod.fnp
    with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace='probe-' + name.replace(' ', '_')) as ctx:
        Ck = np.asarray(mod._shared_old384_mm(fnp.asarray(A), fnp.asarray(B))).astype(np.float64)
        Cp = np.asarray(fnp.matmul(fnp.asarray(A), fnp.asarray(B))).astype(np.float64)
    for tag, C in (('fast kernel', Ck), ('plain matmul', Cp)):
        diff = C - ref
        print('%-34s %-13s max|err|/max|C| = %.2e | frob rel = %.2e | max|err|/median|C| = %.2e' % (name, tag, np.abs(diff).max() / np.abs(ref).max(), np.linalg.norm(diff) / np.linalg.norm(ref), np.abs(diff).max() / np.median(np.abs(ref))), flush=True)
check('start 960x1024 @ 1024x384', J0, (w[c0][:, None] * J0[rows].T).astype(np.float32))
Y1 = (J0.astype(np.float64) @ (w[c0][:, None] * J0[rows].T)).astype(np.float32)
check('Z 2048x960 @ 960x384', np.ascontiguousarray(J1.T), Y1)
Z = (J1.T.astype(np.float64) @ Y1).astype(np.float32)
check('Y2 960x2048 @ 2048x384', J1, (w[c1][:, None] * Z).astype(np.float32))
check('today 960x6144 @ 6144x384', J, (w[:, None] * J[rows].T).astype(np.float32))
