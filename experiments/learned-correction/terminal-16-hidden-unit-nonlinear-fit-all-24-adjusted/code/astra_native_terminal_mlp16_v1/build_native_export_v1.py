from pathlib import Path
import ast,hashlib,json
import numpy as np
D=Path(__file__).resolve().parent
P=D.parent/'astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)=='4d7739a5ad4444be8c20bdd5c5bc86f1cee5536a2c21e956d86dd95e6c83d8d9'
r=json.loads((D/'ALL32_FIT_ACTUAL_RETIREMENT_v1.json').read_text())
assert r['actual_exit_observed'] and r['exit_code']==0 and r['all_owned_PID_PPID_PGID_absent']
meta=json.loads((D/'ALL32_FIT_READY_v1.json').read_text());assert meta['selected_shrink']==.5 and meta['steps']==500
assert sha(meta['map_path'])==meta['map_sha256']
with np.load(meta['map_path'],allow_pickle=False)as z:a={k:z[k] for k in z.files}
pieces=['\n# Frozen TRAIN-only terminal residual MLP; parent physics and state unchanged.\n']
for name,value in [('CENTER',a['center']),('SCALE',a['scale']),('W0',a['w0'].T),('B0',a['b0']),('W1',a['w1'].T),('B1',a['b1'])]:
 pieces.append('_TMLP_'+name+' = fnp.asarray('+repr(value.tolist())+', dtype=fnp.float64)\n')
pieces.append('_TMLP_TARGET_RMS = '+repr(float(a['target_rms']))+'\n')
pieces.append('''_TMLP_SHRINK = 0.5
_TMLP_ON = True
_TMLP_STATS = dict(calls=0, layer=None, state_identity=True)
def _tmlp_normalized(x, state):
    X = fnp.asarray(fnp.concatenate((x, state), axis=1), dtype=fnp.float64)
    Z = (X - _TMLP_CENTER) / _TMLP_SCALE
    H = Z @ _TMLP_W0 + _TMLP_B0
    H = H / (1.0 + fnp.exp(-H))
    Y = (H @ _TMLP_W1 + _TMLP_B1).reshape((-1,))
    return (Y * _TMLP_TARGET_RMS) * _TMLP_SHRINK
_TMLP_PARENT_GNN = gnn_step
def gnn_step(l, x, sigma, W, state, P):
    answer = _TMLP_PARENT_GNN(l, x, sigma, W, state, P)
    if not _TMLP_ON or int(l) != 15:
        return answer
    assert x.shape[1] == 30 and answer[1].shape == (x.shape[0], 64)
    correction = fnp.asarray(sigma, dtype=fnp.float64) * _tmlp_normalized(x, answer[1])
    delta = fnp.asarray(answer[0], dtype=fnp.float64) + correction
    _TMLP_STATS.update(calls=_TMLP_STATS['calls']+1, layer=15, state_identity=True)
    return delta, answer[1]
''')
f=D/'fragment_terminal_mlp_HELD_v1.py';f.open('x').write(''.join(pieces));ast.parse(f.read_text())
c=D/'candidate_QR_terminal_mlp_ON_HELD_v1.py';c.open('xb').write(P.read_bytes()+f.read_bytes())
assert c.read_bytes()[:-len(f.read_bytes())]==P.read_bytes()
print(json.dumps(dict(candidate=str(c),candidate_sha256=sha(c),fragment_sha256=sha(f),map_sha256=meta['map_sha256'])))
