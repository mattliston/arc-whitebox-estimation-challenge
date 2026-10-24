"""Keep orthogonal projections but approximate the future range-finder core.

For Y=G[:,I]=QR, use R inv(G[I,I]) R.T as the next range-finder's
core. The complete atom factors still use Q.T J and Q.T Qold, exactly as
the parent. The terminal core is omitted because it has no consumers.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
P2=HERE.parent
base=Path(sys.argv[1]) if len(sys.argv)>1 else P2/'fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py'
tag=sys.argv[2] if len(sys.argv)>2 else 'candidate_core_nystrom_v1'
src=base.read_text()
node=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'][-1]
fn='\n'.join(src.splitlines()[node.lineno-1:node.end_lineno])+'\n'
assert fn.count('del _jn_d, _jn_rows')==1
fn=fn.replace('del _jn_d, _jn_rows', 'del _jn_d')
old="        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')\n        del unused_R, Y, Omega"
assert fn.count(old)==1
fn=fn.replace(old, '''        _slr_require(_JOIN_NYSTROM, 'Nystrom core requires Nystrom columns')
        _need_core = layer < max(_JOIN_SCHEDULE)
        K = Y[_jn_rows, :].astype(fnp.float64) if _need_core else None
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:
            K = (K + K.T) * 0.5
            R64 = unused_R.astype(fnp.float64)
            core = (R64 @ fnp.linalg.solve(K, R64.T)).astype(fnp.float32)
            core = (core + core.T) * 0.5
            del R64
        del K, unused_R, Y, Omega, _jn_rows''')
old="        joined_core = _shared_old384_mm(Fj, _e2_H(_e2_H(weight[:, None]) * _e2_H(Fj.T)))"
assert fn.count(old)==1
fn=fn.replace(old, '        joined_core = None')
old="            core = _shared_old384_mm(overlap, _e2_H(_shared_old384_mm(state['core'], _e2_H(overlap.T))))\n            core = core + joined_core\n"
assert fn.count(old)==1
fn=fn.replace(old, '')
assert fn.count('            factors, core = Fj, joined_core')==1
fn=fn.replace('            factors, core = Fj, joined_core','            factors = Fj')
old="        state['core'] = _shared_old384_finite(core, 'weighted Gram core')"
assert fn.count(old)==1
fn=fn.replace(old,"        state['core'] = _shared_old384_finite(core, 'Nystrom approximate Gram core') if _need_core else None")
fragment='\n\n# Approximate range-finder core, original orthogonal factor projection.\n'+fn
out=src+fragment
ast.parse(out)
for path,value in [(HERE/(tag+'.py'),out),(HERE/(tag+'_fragment.py'),fragment)]:
    with path.open('x') as f:f.write(value)
record=dict(base=str(base),base_sha256=hashlib.sha256(src.encode()).hexdigest(),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),
            approximation='future basis-selection core only; factor projection unchanged')
with (HERE/(tag+'_manifest.json')).open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
