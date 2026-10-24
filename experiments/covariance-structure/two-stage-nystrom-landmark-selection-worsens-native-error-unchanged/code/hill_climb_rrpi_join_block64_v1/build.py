"""Source-only two-stage64 Gram landmarks; current ranks and final QR unchanged."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
H=D.parent
P=H/'hill_climb_pair_rr_inverse_v1/candidate.py'
PARENT_SHA='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
SUFFIX='''

# Fixed two-stage cohort Gram landmarks; not the failed one-pivot selector.
_JB64_ENABLED = True


def _jb64_columns(J, weight, Qp, core, indices):
    hold = _E2Hold()
    Y = _shared_old384_mm(J, hold(hold(weight[:, None]) * hold(hold(J[indices, :]).T)))
    if Qp is not None:
        Y = Y + hold(_shared_old384_mm(Qp, hold(_shared_old384_mm(core, hold(hold(Qp[indices, :]).T)))))
    return Y


def _jb64_select(J, weight, Qp, core, diagonal, rank):
    hold = _E2Hold()
    if rank not in (320,384) or int(J.shape[0]) < rank:
        raise RuntimeError('Join block64 fixed current-rank domain')
    first = fnp.sort(hold(fnp.argsort(hold(-hold(diagonal.astype(fnp.float64))), kind='stable'))[:64])
    G0 = _jb64_columns(J, weight, Qp, core, first)
    G064 = G0.astype(fnp.float64)
    K0 = G064[first, :]
    K0 = (K0 + hold(K0.T)) * 0.5
    ridge = 1e-6 * hold(fnp.mean(hold(fnp.diag(K0))))
    if not bool(fnp.isfinite(ridge)) or not bool(ridge > 0.0):
        raise RuntimeError('Join block64 positive finite fixed regularizer')
    regularized = K0 + hold(ridge * hold(fnp.eye(64, dtype=fnp.float64)))
    solved = fnp.linalg.solve(regularized, hold(G064.T)).T
    if not bool(fnp.all(hold(fnp.isfinite(solved)))):
        raise RuntimeError('Join block64 nonfinite conditional solve')
    explained = fnp.sum(hold(G064 * solved), axis=1)
    if not bool(fnp.all(hold(fnp.isfinite(explained)))):
        raise RuntimeError('Join block64 nonfinite explained diagonal')
    residual = fnp.maximum(hold(diagonal.astype(fnp.float64)) - explained, 0.0)
    if not bool(fnp.all(hold(fnp.isfinite(residual)))):
        raise RuntimeError('Join block64 nonfinite residual diagonal')
    fnp.put(residual, first, -1.0)
    remaining = fnp.sort(hold(fnp.argsort(hold(-residual), kind='stable'))[:rank-64])
    G1 = _jb64_columns(J, weight, Qp, core, remaining)
    selected = fnp.concatenate((first, remaining))
    order = fnp.argsort(selected, kind='stable')
    indices = selected[order]
    Y = hold(fnp.concatenate((G0, G1), axis=1))[:, order]
    return indices, Y
'''


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def source():
    assert pin(P)['sha256']==PARENT_SHA
    base=P.read_text();tree=ast.parse(base)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push']
    node=next(n for n in reversed(nodes) if '_need_core = layer < max(_JOIN_SCHEDULE)' in ast.get_source_segment(base,n))
    function=ast.get_source_segment(base,node)
    before=("            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])\n"
            "            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))\n"
            "            if old_columns:\n"
            "                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))")
    before += '\n            del _jn_d'
    assert function.count(before)==1 and base.count(function)==1
    enabled=("            if _JB64_ENABLED:\n"
             "                _jn_rows, Y = _jb64_select(J, weight, Qp, state['core'], _jn_d, rank)\n"
             "            else:\n"+'\n'.join('    '+line for line in before.splitlines()[:-1])+'\n            del _jn_d')
    replacement=function.replace(before,enabled,1)
    edits=[dict(before=function,after=replacement)]
    body=base.replace(function,replacement,1);candidate=body+SUFFIX
    assert body.replace(replacement,function,1)==base
    assert candidate.count('_JB64_ENABLED = True')==1
    identity=candidate.replace('_JB64_ENABLED = True','_JB64_ENABLED = False',1)
    compile(candidate,str(D/'candidate.py'),'exec');compile(identity,str(D/'candidate_identity.py'),'exec')
    return candidate,identity,edits


def main():
    candidate,identity,edits=source()
    for name,value in [('candidate.py',candidate),('candidate_identity.py',identity)]:
        with (D/name).open('x') as f:f.write(value)
    prior=pin(H/'hill_climb_join_residual_pivot_v1/closeout.json');r=json.loads(Path(prior['path']).read_text())
    assert r['status']=='CLOSED_SCIENTIFIC_FAILURE' and r['parent']==pin(P)
    record=dict(parent=pin(P),candidate=pin(D/'candidate.py'),identity=pin(D/'candidate_identity.py'),build=pin(__file__),
        edits=edits,suffix=SUFFIX,whole_source_inverse_exact=True,compiled=True,enabled=True,
        first_landmarks=64,ridge_relative_to_mean_diagonal=1e-6,ranks=[384,320,384],joins=[7,10,13],common_rank=256,
        stages='First64 by current raw Gram diagonal; one ridge Schur residual update; remaining rank-64 by residual. Final landmarks and reused Gram columns sorted by physical index.',
        gram_metric_unchanged=True,current_final_QR_core_and_factor_code_unchanged=True,
        identity_original_array_operations_exact_by_source=True,
        cost='Same total raw Gram columns in two batches; added64solve/residual arithmetic and repeated transform/RPC overhead need actual billing and resources.',
        prior_one_pivot_failure=prior,prior_one_pivot_metrics=r['metrics'],
        novelty='Two-stage64 conditional selection, distinct from one-pivot update and star32 projection; no claim of expected improvement.',
        source_only=True,no_model_or_array_jobs=True)
    with (D/'source_delta.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:record[k] for k in ('candidate','identity','build')}))


if __name__=='__main__':main()
