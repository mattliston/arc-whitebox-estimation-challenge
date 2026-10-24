"""TRAIN-only diagnostic of negative-side total CP diagonals, unchanged S30."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
source=(HERE/'native_eval.py').read_text()
marker='class MLP:\n'
assert source.count(marker)==1
instrument='''
_nd_records = []
_nd_original = mod._deadcut_push
def _nd_push(layer, w32, p, previous, newborn, lam, state, alpha):
    pp = _nd_original(layer,w32,p,previous,newborn,lam,state,alpha)
    if layer not in (7,10,14):
        return pp
    P = np.asarray(pp,dtype=np.float64)
    coefficient = np.asarray(lam,dtype=np.float64)
    a = np.asarray(alpha)
    g = np.asarray(mod._E2_LAYER['g'])
    order = np.argsort(-a,kind='stable')
    baseline_kept = min(len(a),int(np.sum(a>=-3.5)+63)//64*64)
    alive_count = min(len(a),int(np.sum(g>=.01)+31)//32*32)
    columns = int(state['columns'])
    inherited = int(newborn[0])
    variance = np.sum(P*P,axis=0)/len(P)
    traces = 3*(P@(coefficient*variance))
    oldtrace = 3*(P[:,:columns]@(coefficient[:columns]*variance[:columns]))
    newborntrace = 3*(P[:,inherited:]@(coefficient[inherited:]*variance[inherited:]))
    entry = dict(rank=rank,layer=layer,columns=P.shape[1],old_columns=columns,newborn_start=inherited,
                 baseline_kept=baseline_kept,alive_count=alive_count,thresholds=[])
    for z in (3.0,2.75,2.5,2.33):
        proposed = min(len(a),int(np.sum(a>=-z)+63)//64*64)
        # This diagnostic preserves every actually selected (padded) live row.
        kept = max(proposed,((alive_count+63)//64)*64)
        extra = order[kept:baseline_kept]
        if not len(extra):
            entry['thresholds'].append(dict(z=z,extra_rows=0));continue
        E=P[extra]
        diag=np.sum((E*E*E)*coefficient,axis=1)
        olddiag=np.sum((E[:,:columns]**3)*coefficient[:columns],axis=1)
        newdiag=np.sum((E[:,inherited:]**3)*coefficient[inherited:],axis=1)
        variants = dict(trace_all=traces[extra],
                        trace_older_exact_newborn=traces[extra]-newborntrace[extra]+newdiag,
                        trace_old_exact_young=oldtrace[extra]+diag-olddiag)
        metrics = {k:float(np.sum((v-diag)**2)/np.sum(diag**2)) for k,v in variants.items()}
        entry['thresholds'].append(dict(z=z,extra_rows=len(extra),preserved_alive=True,
             total_diagonal_rms=float(np.sqrt(np.mean(diag*diag))),
             old_fraction_squared=float(np.sum(olddiag**2)/np.sum(diag**2)),residual_squared_fraction=metrics))
    _nd_records.append(entry)
    (out/'negative_diagonal_probe.json').write_text(json.dumps(dict(
       purpose='TRAIN-only full-PP empirical trace diagnostic; no candidate cost claim',records=_nd_records),indent=2)+'\\n')
    print(json.dumps(dict(event='negative_diagonal_probe',**entry)),flush=True)
    return pp
mod._deadcut_push = _nd_push
'''
source=source.replace(marker,instrument+'\n'+marker)
source+='''
baseline=HERE.parent/'codex_resume_20260926_v1/s30_profile_train2_v1'
for r in ranks:
    assert np.load(out/f'pred_{r}.npy').tobytes()==np.load(baseline/f'pred_{r}.npy').tobytes()
print('ALL_PREDICTIONS_BYTE_IDENTICAL',flush=True)
'''
exec(compile(source,str(HERE/'native_eval.py'),'exec'),globals())
