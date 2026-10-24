"""Passive S31 derived-feature capture; no truth arrays are read."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
protocol_path = HERE / 'response_message_observer_protocol_v1.json'
protocol = json.loads(protocol_path.read_text())
source = (HERE / 'native_eval.py').read_text()
instrument = '''
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == protocol['parent_sha256']
assert ranks in protocol['capture_groups']
_rm_inputs = json.loads((P2/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
assert set(protocol['ranks']).issubset(_rm_inputs['train'])
assert not set(protocol['ranks']).intersection(_rm_inputs['valid'])
del _rm_inputs
_rm_records = []
_rm_state = dict(rank=None, layer=None, history=None)
_rm_feature_original = mod.features
_rm_final_original = mod.final_scalar_features
_rm_step_original = mod.gnn_step
_rm_controls = {}
for _rm_tag in ['both_tail_train2_v6', 'both_tail_train6_v6_routing_control']:
    for _rm_row in map(json.loads, (HERE/_rm_tag/'rows.jsonl').read_text().splitlines()):
        _rm_controls[_rm_row['rank']] = dict(row=_rm_row, tag=_rm_tag)

def _rm_advance(l, aux, w):
    n = len(aux['sigma'])
    if l == 8:
        assert _rm_state['rank'] != rank
        _rm_state.update(rank=rank, layer=7, history=np.zeros((n, 0), dtype=np.float64))
    assert _rm_state['rank'] == rank and _rm_state['layer'] == l-1
    previous = _rm_state['history']
    if previous.shape[1]:
        gate = np.asarray(aux['gate'], dtype=np.float64)
        previous = (np.asarray(w, dtype=np.float64).T @ previous) * gate[:, None]
    _rm_state.update(layer=l, history=previous)

def _rm_features(l, m, C, K21, aux, w):
    x = _rm_feature_original(l, m, C, K21, aux, w)
    if 8 <= l <= 14:
        _rm_advance(l, aux, w)
        sigma = np.asarray(aux['sigma'], dtype=np.float64)
        alpha = np.asarray(aux['alpha'], dtype=np.float64)
        pdf = np.asarray(aux['pdf'], dtype=np.float64)
        phi = np.column_stack((pdf, alpha*pdf, (alpha*alpha-1)*pdf))
        R = np.asarray(aux['pre_cov'], dtype=np.float64) / (sigma[:, None]*sigma[None, :])
        T = np.asarray(K21, dtype=np.float64) / ((sigma*sigma)[:, None]*sigma[None, :])
        np.fill_diagonal(R, 0)
        np.fill_diagonal(T, 0)
        rootn = float(len(sigma))**0.5
        received = np.concatenate((R@phi/rootn, (R*R)@phi, T@phi/rootn, T.T@phi/rootn), axis=1)
        born = (received[:, :, None] * phi[:, None, :]).reshape(len(sigma), 36) * sigma[:, None]
        assert np.isfinite(born).all()
        _rm_state['history'] = np.concatenate((_rm_state['history'], born), axis=1)
    return x

def _rm_final(l, m, aux, w):
    x = _rm_final_original(l, m, aux, w)
    assert l == 15
    _rm_advance(l, aux, w)
    return x

def _rm_step(l, x, sigma, w, state, P):
    result = _rm_step_original(l, x, sigma, w, state, P)
    if l == 15:
        assert _rm_state['rank'] == rank and _rm_state['layer'] == 15
        sd = np.asarray(sigma, dtype=np.float64).copy()
        control = np.concatenate((np.asarray(x, dtype=np.float64), np.asarray(result[1], dtype=np.float64)), axis=1)
        history = _rm_state['history'] / sd[:, None]
        assert control.shape == (1024, 94) and history.shape == (1024, 252)
        assert np.isfinite(control).all() and np.isfinite(history).all() and (sd > 0).all()
        path = out/f'features_rank{rank:04d}.npz'
        with path.open('xb') as f:
            np.savez(f, control=control, history=history, sigma=sd)
        _rm_records.append(dict(rank=rank, path=path.name, bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert sum(rec['bytes'] for rec in _rm_records) < 100_000_000
        _rm_state['history'] = None
    return result

mod.features = _rm_features
mod.final_scalar_features = _rm_final
mod.gnn_step = _rm_step
'''
marker = 'class MLP:\n'
assert source.count(marker) == 1
source = source.replace(marker, instrument + '\n' + marker)
start = source.index('        truth=np.load(')
end = source.index('        np.save(out/', start)
source = source[:start] + '''        reference = _rm_controls.get(rank)
        exact = bill_exact = None
        if reference is not None:
            base = np.load(HERE/reference['tag']/f'pred_{rank}.npy')
            exact = bool(np.array_equal(y, base))
            bill_exact = flops_used == reference['row']['flops']
            assert exact and bill_exact, (rank, exact, bill_exact, flops_used)
            del base
        row = dict(rank=rank, flops=flops_used, independent_control_available=reference is not None,
                   all16_bytes_exact=exact, bill_exact=bill_exact, wall_s=seconds, cpu_s=cpu,
                   truth_accessed=False)
''' + source[end:]
source = source.replace('del weights,y,truth', 'del weights,y')
source += '''
assert [r['rank'] for r in _rm_records] == ranks
with (out/'capture_manifest.json').open('x') as f:
    json.dump(dict(purpose=protocol['purpose'], records=_rm_records, truth_accessed=False,
                   protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                   capture_script_sha256=hashlib.sha256((HERE/'capture_response_messages_v1.py').read_bytes()).hexdigest(),
                   diagnostic_copies_not_deployable=True), f, indent=2)
print(json.dumps(dict(captures=len(_rm_records), bytes=sum(r['bytes'] for r in _rm_records))), flush=True)
'''
assert 'truth=np.load(' not in source
exec(compile(source, str(HERE/'native_eval.py'), 'exec'), globals())
