"""[opus-ybank] v2: + cohort pair split (common c / difference h) | measurement-only probe of the young (newborn) bank on ONE network. Same setup/predict emulation as
lead/runs/opus_f384_v1/diag/diag_judge_v2.py (PB_ALL prebind, unlimited setup wall, flopscope BudgetContext 2**41).
Read-only instrumentation: wrappers around the module globals _deadcut_push (whole per-layer bank push) and
_push_bank_shared_newborn (young + newborn push) take PLAIN-numpy copies and read the running bill (ctx.flops_used);
no flopscope op is added, so predictions and FLOPs must equal the uninstrumented run (checked against a reference).
usage (OpenBLAS python, BLAS threads 1, PB_ALL=1): ybank_probe_v1.py <candidate.py> <seed> <out.json> [<reference diag json>]"""
import hashlib, json, math, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh'))
import loader
cand, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
ref = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
assert not out.exists() and os.environ.get('PB_ALL') == '1'
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_ybank'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9

def used():
    c = _budget.get_active_budget(); return int(c.flops_used) if c is not None else 0
REC, CUR = {}, [None]
P_DEAD, P_YOUNG = mod._deadcut_push, mod._push_bank_shared_newborn
def deadcut(layer, w32, p, previous, newborn, lam, state, a):
    CUR[0] = int(layer); f0 = used(); old0 = int(state['columns']); births = (int(np.asarray(newborn[1]).shape[0]) if newborn is not None else 0)
    pp = P_DEAD(layer, w32, p, previous, newborn, lam, state, a)
    f1 = used()
    r = REC.setdefault(int(layer), dict(young_push_flops=0, young_calls=0))
    old = int(state['columns']); A = np.array(pp, dtype=np.float32, copy=True); L = np.array(lam, dtype=np.float32, copy=True)
    r.update(births_in=births, old_before=old0, total_push_flops=f1 - f0, rows=int(A.shape[0]), K=int(A.shape[1]), old=old, lam_len=int(L.shape[0]),
             prev_rows=int(np.asarray(previous).shape[0]), Y=A[:, old:], lam_y=L[old:] if L.shape[0] == A.shape[1] else None)
    return pp
def young(w, p, previous, newborn):
    f0 = used(); res = P_YOUNG(w, p, previous, newborn); f1 = used()
    r = REC.setdefault(CUR[0], dict(young_push_flops=0, young_calls=0))
    r['young_push_flops'] += f1 - f0; r['young_calls'] += 1
    r['young_in_cols'] = int(np.asarray(p).shape[1]); r['young_out_shape'] = list(np.asarray(res).shape)
    return res
mod._deadcut_push, mod._push_bank_shared_newborn = deadcut, young

est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed); truth = loader.truth(seed, check_weights=False)['mc_post']
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
t0 = time.monotonic()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
total = int(ctx.summary_dict()['flops_used']); wall = time.monotonic() - t0

def eff(e):   # energies (nonnegative), descending -> ranks at 99% / 99.9% / 99.99%
    e = np.sort(np.maximum(e, 0))[::-1]; c = np.cumsum(e) / max(e.sum(), 1e-300)
    return [int(np.searchsorted(c, q) + 1) for q in (0.99, 0.999, 0.9999)]
cohorts = []
for l in sorted(REC):
    r = REC[l]
    if 'births_in' not in r: continue
    if r['births_in']: cohorts.append(2 * r['births_in'])
    joined = r['old'] - r['old_before']
    while joined > 0: joined -= cohorts.pop(0)
    assert joined == 0, ('cohort join mismatch', l)
    r['cohorts'] = list(cohorts)
layers = []
for l in sorted(REC):
    r = REC[l]; Y = r.pop('Y', None); ly = r.pop('lam_y', None)
    if Y is None or Y.shape[1] == 0:
        layers.append(dict(layer=l, **r, young_cols=0)); continue
    Yd = Y.astype(np.float64)
    plain = np.linalg.eigvalsh(Yd @ Yd.T)                              # = singular values^2 of Y
    row = dict(layer=l, **r, young_cols=int(Y.shape[1]), rank_cap=int(min(Y.shape)), r_plain=eff(plain))
    if ly is not None:
        lw = ly.astype(np.float64)
        Yw = Yd * np.sqrt(np.abs(lw))[None, :]
        row['r_abslam'] = eff(np.linalg.eigvalsh(Yw @ Yw.T))           # |lambda|-weighted column energy
        ev = np.linalg.eigvalsh((Yd * lw[None, :]) @ Yd.T)                 # signed covariance Y diag(lam) Y^T
        row['r_signed_cov'] = eff(np.abs(ev)); row['cov_pos_neg_trace'] = [float(ev[ev > 0].sum()), float(-ev[ev < 0].sum())]
        Y2 = Yd * Yd                                                        # squared columns, as the read's left side
        row['r_squared_cols_abslam'] = eff(np.linalg.eigvalsh((Y2 * np.sqrt(np.abs(lw))[None, :]) @ (Y2 * np.sqrt(np.abs(lw))[None, :]).T))
    if ly is not None and sum(r.get('cohorts', [])) == Y.shape[1]:
        Cs, Hs, i = [], [], 0
        for size in r['cohorts']:
            b = size // 2; t, s_ = Yd[:, i:i + b], Yd[:, i + b:i + size]; a = lw[i:i + b]
            assert np.array_equal(ly[i:i + b], -ly[i + b:i + size]), ('pair weights', l)
            Cs.append(((t + s_) / 2) * np.sqrt(np.abs(a))[None, :]); Hs.append(((t - s_) / 2) * np.sqrt(np.abs(a))[None, :]); i += size
        Cm, Hm = np.concatenate(Cs, axis=1), np.concatenate(Hs, axis=1)
        gc, gh = Cm @ Cm.T, Hm @ Hm.T
        row['r_common'] = eff(np.linalg.eigvalsh(gc)); row['r_diff'] = eff(np.linalg.eigvalsh(gh))
        row['energy_common_diff'] = [float(np.trace(gc)), float(np.trace(gh))]
        ew, U = np.linalg.eigh(Yd @ Yd.T); U = U[:, ::-1]
        for q, rr in zip(('99', '999'), row['r_plain'][:2]):
            Ur = U[:, :rr]; row[f'h_energy_kept_at_r{q}'] = float(np.sum((Ur.T @ Hm) ** 2) / np.trace(gh)); row[f'c_energy_kept_at_r{q}'] = float(np.sum((Ur.T @ Cm) ** 2) / np.trace(gc))
    layers.append(row)
res = dict(candidate=str(cand), sha256=hashlib.sha256(cand.read_bytes()).hexdigest(), seed=seed, total_flops=total, predict_wall_s=wall,
           pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest(), mse_final=float(np.mean((y[15] - truth[15])**2)), layers=layers)
if ref is not None:
    rj = json.loads(ref.read_text())['rows'][0]; rp = np.load(str(ref).replace('.json', '_pred.npz'))[str(seed)]
    res['reference'] = dict(path=str(ref), flops_equal=(rj['flops'] == total), pred_bytes_equal=(rp.tobytes() == y.astype(np.float32).tobytes()), ref_flops=rj['flops'])
with out.open('x') as f: json.dump(res, f, indent=1)
print(json.dumps(dict(done=True, total=total, reference=res.get('reference'))), flush=True)
