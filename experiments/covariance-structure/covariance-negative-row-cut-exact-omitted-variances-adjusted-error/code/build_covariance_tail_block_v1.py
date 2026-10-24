"""Append a localized covariance consumer approximation to immutable S31."""
from pathlib import Path
import ast
import hashlib
import json
HERE=Path(__file__).resolve().parent
protocol_path=HERE/'covariance_tail_block_protocol_v1.json'
protocol=json.loads(protocol_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=HERE/protocol['parent'];assert sha(parent)==protocol['parent_sha256']
cert=json.loads((HERE/'covariance_tail_block_certificate_v1.json').read_text())
assert cert['passed'] and cert['protocol_sha256']==sha(protocol_path)
source=parent.read_text();lines=source.splitlines()
node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_flms_cov_star'][-1]
body='\n'.join(lines[node.lineno-1:node.end_lineno])
body=body.replace('    C32 = C.astype(fnp.float32)\n','')
old='    first = _mm(C32, w)';assert body.count(old)==1
body=body.replace(old,'    Cpre = _ctb_push(w, C, diagonal_only=False)')
old='    cov = _sym_upper(wt, first)\n    Cpre = cov.astype(fnp.float64)';assert body.count(old)==1
body=body.replace(old,'')
helper='''
# Local covariance tail block; all retained and omitted diagonal work is paid.
_CTB_STATS = dict(calls=0, cuts=0, kept=[])
_CTB_ORIGINAL_DIAGONAL = _flms_covariance_diagonal

def _ctb_push(w, C, diagonal_only=False):
    n = int(w.shape[0])
    a = _E2_LAYER['a']
    count = int(fnp.nonzero(a >= -2.5)[0].shape[0])
    kept = min(n, max(64, ((count + 63) // 64) * 64))
    _CTB_STATS['calls'] += 1
    _CTB_STATS['kept'].append(kept)
    if kept == n:
        if diagonal_only:
            return _CTB_ORIGINAL_DIAGONAL(w, C)[0]
        first = _mm(C.astype(fnp.float32), w)
        return _sym_upper(w.T, first).astype(fnp.float64)
    _CTB_STATS['cuts'] += 1
    ordered = fnp.argsort(-a, kind='stable')
    rows = fnp.sort(ordered[:kept])
    tail = fnp.sort(ordered[kept:])
    wr, wd = w[rows, :], w[tail, :]
    ck = C[rows[:, None], rows[None, :]].astype(fnp.float32)
    dt = fnp.diag(C)[tail].astype(fnp.float32)
    first = _mm(ck, wr)
    if diagonal_only:
        live = fnp.sum(wr.astype(fnp.float64) * first.astype(fnp.float64), axis=0)
        dead = fnp.sum((wd.astype(fnp.float64) * wd.astype(fnp.float64)) * dt.astype(fnp.float64)[:, None], axis=0)
        return live + dead
    live = _sym_upper(wr.T, first).astype(fnp.float64)
    dead = _sym_upper(wd.T, wd * dt[:, None]).astype(fnp.float64)
    return live + dead

def _flms_covariance_diagonal(w, C):
    v = _ctb_push(w, C, diagonal_only=True)
    return v, v
'''
fragment='\n\n'+helper+'\n\n'+body+'\n'
ast.parse(fragment)
candidate=HERE/'candidate_covariance_tail_block_v1.py'
with candidate.open('x') as f:f.write(source+fragment)
assert candidate.read_bytes().startswith(parent.read_bytes())
record=dict(candidate=candidate.name,candidate_sha256=sha(candidate),parent_sha256=sha(parent),
            protocol_sha256=sha(protocol_path),certificate_sha256=sha(HERE/'covariance_tail_block_certificate_v1.json'),
            original_prefix_bytes_preserved=True)
with (HERE/'covariance_tail_block_build_v1.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
