"""Historical mechanism proxy only: no SDK, labels, training, or candidate edits."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1]
DATA = RUNS / 'opus_fix336687_v1/joinsketch'

def energy(f, lam):
    return float(np.sum(np.sum(np.square(f), axis=0, dtype=np.float64)
                        * np.square(lam.astype(np.float64)), dtype=np.float64))

def norm(a):
    return float(np.sqrt(np.sum(np.square(a.astype(np.float64)), dtype=np.float64)))

def probe(network):
    started = time.monotonic()
    z = np.load(DATA / f'join{network}_L13.npz')
    with np.load(DATA / f'join{network}_L10.npz') as earlier:
        prefix = int(earlier['old_columns'])
    J, lam, oldlam, Qp, F, core = (z[k] for k in ('J', 'lam', 'lam_old', 'Qp', 'F', 'core'))
    rank = Qp.shape[1]
    beta = energy(F[:, prefix:], oldlam[prefix:]) / energy(F, oldlam)
    previous = z['previous']
    W = z['w32'][previous].T[::16].copy()
    assert W.shape[0] == 64 and rank == 384
    X = np.concatenate((Qp @ F, J), axis=1)
    weights_all = np.concatenate((oldlam, lam))
    direct = W @ X
    target = (np.square(direct) * weights_all[None, :]) @ X.T
    target64 = target @ W.T
    target_norm = norm(target)
    target64_norm = norm(target64)
    JJ = np.square(J)
    weight = np.square(lam)
    colscore = np.sum(JJ, axis=0) * weight
    cols = np.argsort(-colscore.astype(np.float64), kind='stable')[:1536]
    J1 = J[:, cols]
    dnew = JJ @ weight
    dold = np.sum((Qp @ core) * Qp, axis=1)
    del JJ
    results = []
    for label, scale in [('attenuated_beta', beta), ('full_stored_core', 1.0)]:
        d = dnew + np.float32(scale) * dold
        selected = np.sort(np.argsort(-d.astype(np.float64), kind='stable')[:rank])
        Y = J1 @ (weight[cols, None] * J1[selected].T)
        Y += Qp @ ((np.float32(scale) * core) @ Qp[selected].T)
        Qn, _ = np.linalg.qr(Y, mode='reduced')
        factors = Qn.T @ X
        XP = Qn @ factors
        propagated = W @ XP
        plane = (np.square(propagated) * weights_all[None, :]) @ XP.T
        error = norm(plane - target)
        error64 = norm(plane @ W.T - target64)
        results.append(dict(label=label, inherited_core_scale=scale,
                            selected_rows=selected.tolist(),
                            cp_plane_error=error, cp_plane_relative_error=error/target_norm,
                            propagated64_error=error64,
                            propagated64_relative_error=error64/target64_norm))
        del Y, Qn, factors, XP, propagated, plane
        gc.collect()
    a, b = results
    return dict(network=network, prefix=prefix, beta=beta,
                total_old_columns=int(F.shape[1]), joined_columns=int(J.shape[1]),
                basis_rank=rank, kc=1536, results=results,
                full_over_attenuated_cp_plane_error=b['cp_plane_error']/a['cp_plane_error'],
                full_over_attenuated_propagated64_error=b['propagated64_error']/a['propagated64_error'],
                selected_row_changes=len(set(a['selected_rows']) ^ set(b['selected_rows'])) // 2,
                elapsed_s=time.monotonic()-started)

rows = []
for network in (35, 200):
    rows.append(probe(network))
    print(json.dumps({k: v for k, v in rows[-1].items() if k != 'results'}), flush=True)
    gc.collect()
result = dict(status='HISTORICAL_MECHANISM_PROXY_ONLY', source='pre-JSK historical join snapshots',
              operation='L13 plain Nystrom encoder; infer inherited L10 attenuation from historical full-F weighted energy',
              current_source_candidate_tested=False, labels_used=False, fitted_parameters=False,
              caveats=['Historical full-core saved states differ from actual95c states.',
                       'Both variants use all saved F columns and identical kc1536; only old-core amplitude and induced row selection differ.',
                       'A favorable proxy is mechanism support only, not a native accuracy or adjusted-score gain.'],
              rows=rows, maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(HERE/'RESULT_v1.json').write_text(json.dumps(result, indent=2)+'\n')
