"""Fixed rank16/32 ceilings on the accepted TRAIN16 saved old read; no model/truth."""
from pathlib import Path
import hashlib, json, sys
import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
P = O / 'lead/runs/astra_read_column_oracle_v1'
sys.path.insert(0, str(P))
from inspect_fixed_band_oracle_HELD_v1 import basis, zero_diag, energies
C = O / 'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
out = D / 'TRAIN16_RANK16_32_ORACLE_v1.json'
assert not out.exists()
rows, pins, pooled = [], {}, {}
for path in sorted(C.glob('L*.npz')):
    with np.load(path, allow_pickle=False) as z:
        if 'old_inner' not in z:
            continue
        H = z['old_inner'].astype(float)
        B = z['old_right'].astype(float)
        ids = z['rows'].astype(int)
        cols = z['cols'].astype(int)[:len(B)]
        v, mu, g = [z[k].astype(float) for k in ('vc', 'mu', 'gate')]
    pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    d = np.exp(-.5 * mu[ids]**2 / v[ids]) / np.sqrt(2*np.pi*v[ids])
    Y = H @ B.T
    W = .5 * d[:, None] * Y * g[cols][None, :]
    uh, sh, _ = np.linalg.svd(H, full_matrices=False)
    uw, sw, _ = np.linalg.svd(W, full_matrices=False)
    q0 = basis(v[ids], mu[ids])
    e0 = (H - q0 @ (q0.T @ H)) @ B.T
    zero_diag(e0, ids, cols)
    center = ((3*len(ids)+255)//256)*64-2
    ep = e0.copy(); ep[:center] = 0
    cases = {
        'old_full_energy': energies(Y, d, g[cols], ids, cols, len(mu)),
        'vc_mu_global': energies(e0, d, g[cols], ids, cols, len(mu)),
        'protected_outer_quarter': energies(ep, d, g[cols], ids, cols, len(mu)),
    }
    lower = {}
    for rank in (16, 32):
        q = uh[:, :rank]
        E = (H - q @ (q.T @ H)) @ B.T
        zero_diag(E, ids, cols)
        cases[f'inner_svd_{rank}'] = energies(E, d, g[cols], ids, cols, len(mu))
        # This is the optimal directed weighted-output rank-r approximation
        # BEFORE the exact diagonal overwrite, not an optimal symmetric or
        # diagonal-excluded objective. It allows an oracle weighted row range.
        qw = uw[:, :rank]
        Ew = W - qw @ (qw.T @ W)
        zero_diag(Ew, ids, cols)
        full = np.zeros((len(mu), len(mu)))
        full[np.ix_(ids, cols)] = Ew
        sym = full + full.T
        cases[f'weighted_output_svd_{rank}'] = dict(
            weighted_directed=float(np.sum(Ew*Ew)),
            leading_symmetric_covariance=float(np.sum(sym*sym)))
        lower[str(rank)] = dict(
            inner_relative_error2=float(np.sum(sh[rank:]**2)/np.sum(sh**2)),
            weighted_output_relative_error2_before_diagonal=float(np.sum(sw[rank:]**2)/np.sum(sw**2)))
    for name, vals in cases.items():
        for key, value in vals.items():
            pooled.setdefault(name, {}).setdefault(key, 0.)
            pooled[name][key] += value
    rows.append(dict(layer=int(path.stem[1:]),H_shape=list(H.shape),B_shape=list(B.shape),
                     exact_center=center,cases=cases,spectral=lower))
    print(path.stem, json.dumps(lower), flush=True)
young = json.loads((P/'TRAIN16_FIXED_BAND_ORACLE_v1.json').read_text())['pooled']['incumbent_QR']
ratios = {}
for name, vals in pooled.items():
    if 'svd' not in name:
        continue
    ratios[name] = {key: dict(
        versus_global_rank2=vals[key]/pooled['vc_mu_global'][key],
        versus_failed_protected_outer=vals[key]/pooled['protected_outer_quarter'][key],
        versus_incumbent_young_tail=vals[key]/young[key],
        relative_old_energy=vals[key]/pooled['old_full_energy'][key])
        for key in ('weighted_directed','leading_symmetric_covariance')}
doc = dict(status='FIXED_TWO_RANK_OFFLINE_ORACLE',ranks=[16,32],per_layer=rows,
    pooled=pooled,ratios=ratios,incumbent_young_tail=young,evidence_sha256=pins,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limits=[
        'Only accepted saved TRAIN16 old_inner and retained-prefix old_right. No truths, network calls, new captures or rank sweep.',
        'Inner SVD is unavailable before the old inner; weighted-output SVD is an even stronger nondeployable output oracle.',
        'Weighted-output SVD optimizes directed error before zeroing the exact scalar diagonal. Symmetric covariance error is evaluated, not globally optimized.',
        'Leading covariance weighting is .5*pdf(alpha)/sigma on rows and Gaussian gate on columns; negative read tail and nonlinear future feedback are not captured.',
        'Comparison with failed protected-outer error is an operator-risk benchmark, not a prediction of terminal MSE recovery.'])
out.write_text(json.dumps(doc,indent=2)+'\n')
print('POOLED',json.dumps(ratios),flush=True)
print('SHA256',hashlib.sha256(out.read_bytes()).hexdigest())
