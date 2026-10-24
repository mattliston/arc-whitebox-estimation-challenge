"""Fixed saved-array test: cheaper JSK with tail diagonal or stratified energy.

No labels or parameter fitting. Full signed factors/read retained. Historical
pre-JSK dumps are a screening instrument, not current native score evidence.
"""
import gc
import json
import resource
import time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
LAB = D.parent / 'opus_fix336687_v1/joinsketch'
OUT = D / 'JSK_TAIL_COMPLETION_RESULT_v1.json'
assert not OUT.exists()
started = time.monotonic()
results = []
for rank in (35, 200):
    for layer, baseline_kc, small_kc in ((7, 2048, 1024), (10, 1536, 768), (13, 1536, 768)):
        z = np.load(LAB / f'join{rank}_L{layer}.npz')
        J = z['J'].astype('f'); lam = z['lam'].astype('f'); w = lam * lam
        old = int(z['old_columns']); r = 384
        young_diag = (J * J) @ w
        if old:
            Qp, core, F = (z[k].astype('f') for k in ('Qp', 'core', 'F'))
            Qc = Qp @ core
            diag = young_diag + np.sum(Qc * Qp, axis=1)
            X = np.concatenate((Qp @ F, J), axis=1)
        else:
            Qp = Qc = core = F = None
            diag = young_diag.copy(); X = J.copy()
        rows = np.sort(np.argsort(-diag.astype('d'), kind='stable')[:r])
        old_Y = Qc @ Qp[rows].T if old else np.zeros((J.shape[0], r), dtype='f')
        norm2 = np.sum(J * J, axis=0)
        energies = (norm2 * w).astype('d')
        order = np.argsort(-energies, kind='stable')
        L = np.concatenate((z['lam_old'].astype('d'), lam.astype('d')))
        W = z['w32'][z['previous']].T[::16].astype('d')
        true_push = W @ X.astype('d')
        truth = (true_push * true_push * L) @ true_push.T
        true_k3 = np.diag(truth).copy()
        np.fill_diagonal(truth, 0.)
        arms = {}
        for arm in ('baseline', 'half_heavy', 'half_diag_tail', 'half_energy_strata'):
            kc = baseline_kc if arm == 'baseline' else small_kc
            cols = order[:kc]
            ww = w[cols]
            if arm == 'half_energy_strata':
                nh = kc // 2; nt = kc - nh
                heavy = order[:nh]; tail = order[nh:]
                cumulative = np.cumsum(energies[tail])
                mass = cumulative[-1] / nt
                pos = np.searchsorted(cumulative, (np.arange(nt) + .5) * mass)
                chosen = tail[pos]
                cols = np.concatenate((heavy, chosen))
                ww = np.concatenate((w[heavy], (mass / norm2[chosen]).astype('f')))
            Jc = J[:, cols]
            Y = Jc @ (ww[:, None] * Jc[rows].T)
            if arm == 'half_diag_tail':
                omitted = np.maximum(young_diag - (Jc * Jc) @ ww, 0.)
                Y[rows, np.arange(r)] += omitted[rows]
            Y = Y + old_Y
            arm_metrics = {}
            for noise in (0., 4e-6):
                if noise:
                    rng = np.random.default_rng(17100 + rank * 100 + layer)
                    Yn = Y + (noise * np.max(np.abs(Y)) * rng.standard_normal(Y.shape)).astype('f')
                else:
                    Yn = Y
                Q = np.linalg.qr(Yn)[0].astype('d')
                pushed = (W @ Q) @ (Q.T @ X.astype('d'))
                plane = (pushed * pushed * L) @ pushed.T
                k3_error = float(np.sum((np.diag(plane) - true_k3) ** 2))
                np.fill_diagonal(plane, 0.)
                cp_error = float(np.sum((plane - truth) ** 2))
                arm_metrics[str(noise)] = dict(cp_squared_error=cp_error, k3_squared_error=k3_error)
            arms[arm] = dict(kc=kc, unique_columns=int(len(np.unique(cols))), metrics=arm_metrics)
        row = dict(rank=rank, layer=layer, m=int(J.shape[0]), K=int(J.shape[1]), arms=arms)
        results.append(row)
        print(json.dumps({"rank":rank,"layer":layer,"cp_ratio_vs_baseline":{
            name: {eps: val['cp_squared_error'] / arms['baseline']['metrics'][eps]['cp_squared_error']
                   for eps,val in data['metrics'].items()} for name,data in arms.items()}}), flush=True)
        del z, J, X, Qp, Qc, core, F, Q, pushed, plane, true_push, truth, old_Y, Jc, Y, Yn
        gc.collect()
result = dict(rows=results, elapsed_s=time.monotonic()-started,
              maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              no_truth_labels=True, no_sdk=True,
              caveat='Historical fixed join states; proxies are not native endpoint or adjusted-score measurements.')
with OUT.open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({'saved':str(OUT),'elapsed_s':result['elapsed_s'],'maxrss_bytes':result['maxrss_bytes']}))
