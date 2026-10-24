from pathlib import Path
import ast
import hashlib
import json
HERE=Path(__file__).resolve().parent
P2=HERE.parent
parent=HERE/'candidate_old_both_tail_trace_bounded_v6.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent)=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
source=parent.read_text()
old='                m, aux = nonlinear_scalar(mu, v, k3, _e2_H(_e2_H(radial * v) * v))'
new='''                _terminal_k4 = _e2_H(_e2_H(radial * v) * v)
                if _S19_K4BAND is not None:
                    _terminal_k4 = _e2_H(_terminal_k4 * _S19_K4BAND[2])
                m, aux = nonlinear_scalar(mu, v, k3, _terminal_k4)'''
assert source.count(old)==1
body=source.replace(old,new)
assert body.replace(new,old)==source
ast.parse(body)
tree=ast.parse(source)
assert len([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nonlinear_scalar'])==1
training=P2/'fable_covhunter_v1/s6bcard_joint2t.py'
assert "cd.hooks['k4_15'] = lambda l, k4, ctx: k4 * D['k4c']" in training.read_text()
paths=[training,P2/'fable_physicist_v1/refit6b/s6bcard_loop5js.py',P2/'fable_porter_v1/build_s19_dials.py']
dest=HERE/'candidate_terminal_k4_band_v1.py'
with dest.open('x') as f:f.write(body)
rec=dict(parent_sha256=sha(parent),candidate_sha256=sha(dest),candidate=str(dest),
    whole_source_inverse_exact=True,source_evidence=[dict(path=str(p),sha256=sha(p)) for p in paths],
    final_band=0.9595022755753744,status='Frozen for native TRAIN2; no accuracy result')
with (HERE/'terminal_k4_band_manifest_v1.json').open('x') as f:json.dump(rec,f,indent=2)
print(json.dumps(rec))
