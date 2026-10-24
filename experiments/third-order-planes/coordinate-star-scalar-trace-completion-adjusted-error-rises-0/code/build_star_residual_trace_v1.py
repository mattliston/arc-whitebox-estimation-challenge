"""Build the sealed scalar star-residual trace correction on exact S31."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
protocol_path=HERE/'star_residual_trace_protocol_v1.json'
protocol=json.loads(protocol_path.read_text())
parent=HERE/protocol['parent']
assert hashlib.sha256(parent.read_bytes()).hexdigest()==protocol['parent_sha256']
cert=json.loads((HERE/'star_residual_trace_certificate_v1.json').read_text())
assert cert['passed'] and cert['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
source=parent.read_text(); lines=source.splitlines(); tree=ast.parse(source)
names=['_flms_cov_star','_flms_star_diagonal']
bodies={}
for name in names:
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    bodies[name]='\n'.join(lines[node.lineno-1:node.end_lineno])
helper='''
_LRST_STATS = dict(calls=0)

def _lrst_scalar_trace(w, A32, Q, B):
    _LRST_STATS['calls'] += 1
    n = int(A32.shape[0])
    assert n == 1024 and int(w.shape[0]) == n
    q = Q.astype(fnp.float64)
    b = B.astype(fnp.float64)
    residual_sum = fnp.sum(A32, axis=1, dtype=fnp.float64) - q @ fnp.sum(b, axis=1)
    residual_diag = fnp.diag(A32).astype(fnp.float64) - fnp.sum(q * b.T, axis=1)
    trace3 = residual_sum + 2 * residual_diag
    w64 = w.astype(fnp.float64)
    norm2 = fnp.sum(w64 * w64, axis=0)
    return (norm2 / float(n + 2)) * (w64.T @ trace3)
'''
old='    diagonal = fnp.sum(P1 * T2, axis=1, dtype=fnp.float64)'
assert bodies[names[0]].count(old)==1
bodies[names[0]]=bodies[names[0]].replace(old,old+'\n    diagonal = diagonal + _lrst_scalar_trace(w, A32, Q, B)')
old='    Q, B = _lr_factor(A.astype(fnp.float32))'
assert bodies[names[1]].count(old)==1
bodies[names[1]]=bodies[names[1]].replace(old,'    A32 = A.astype(fnp.float32)\n    Q, B = _lr_factor(A32)')
old='    return fnp.sum((pushed * wt) * wt, axis=1, dtype=fnp.float64)'
assert bodies[names[1]].count(old)==1
bodies[names[1]]=bodies[names[1]].replace(old,'    result = fnp.sum((pushed * wt) * wt, axis=1, dtype=fnp.float64)\n    return result + _lrst_scalar_trace(w, A32, Q, B)')
fragment='\n\n# Fixed scalar-read completion of the discarded star trace.\n'+helper+'\n\n'+'\n\n'.join(bodies.values())+'\n'
ast.parse(fragment)
candidate=HERE/'candidate_star_residual_trace_v1.py'
with candidate.open('x') as fp: fp.write(source+fragment)
record=dict(parent_sha256=protocol['parent_sha256'],candidate=candidate.name,
            candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            protocol_sha256=cert['protocol_sha256'],modified_functions=names,
            original_prefix_bytes_preserved=candidate.read_bytes()[:parent.stat().st_size]==parent.read_bytes())
assert record['original_prefix_bytes_preserved']
with (HERE/'star_residual_trace_build_v1.json').open('x') as fp: json.dump(record,fp,indent=2)
print(json.dumps(record))
