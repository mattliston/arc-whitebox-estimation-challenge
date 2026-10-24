"""Version S29 with target-free interpolatory Nyström joins."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
P2=HERE.parent
base=P2/'fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py'
src=base.read_text()
nodes=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push']
node=nodes[-1]
fn='\n'.join(src.splitlines()[node.lineno-1:node.end_lineno])+'\n'
assert fn.count('del _jn_d, _jn_rows')==1
fn=fn.replace('del _jn_d, _jn_rows','del _jn_d')
start=fn.index("        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')")
end=fn.index('        old_columns = stop',start)
fn=fn[:start]+'''        _slr_require(_JOIN_NYSTROM, 'interpolation requires Nystrom columns')
        # G_II is the same selected Gram block; solving in float64 avoids
        # magnifying solve roundoff. The range-finder products remain metered.
        K = Y[_jn_rows, :].astype(fnp.float64)
        K = (K + K.T) * 0.5
        Qn = fnp.linalg.solve(K, Y.T.astype(fnp.float64)).T.astype(fnp.float32)
        core = K.astype(fnp.float32)
        del K, Y, Omega
        _shared_old384_finite(Qn, 'interpolation basis')
        state['rank_new'] = rank
        Fj = J[_jn_rows, :]
        if old_columns:
            overlap = Qp[_jn_rows, :]
            factors = _slr_join_factors(layer, overlap, state, Fj)
            del overlap
        else:
            factors = Fj
        joined_core = None
        del _jn_rows
''' +fn[end:]
ranks=tuple(int(x) for x in sys.argv[1].split('-'))
assert len(ranks)==3 and all(r%32==0 and 64<=r<=512 for r in ranks)
fragment='\n\n# Interpolatory Nystrom research join, version 1, 2026-09-26.\n'+fn
fragment+='\n_R320_RULE = '+repr(dict(zip((7,10,13),ranks)))+'\n'
name='candidate_interpolation_'+sys.argv[1]+'_v1'
if max(ranks)>384:
    nodes=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_slr_virtual_rotation_envelope']
    n=nodes[-1]
    ef='\n'.join(src.splitlines()[n.lineno-1:n.end_lineno])+'\n'
    assert ef.count('384.0')==1
    fragment+='\n'+ef.replace('384.0',repr(float(max(ranks))))
out=src+fragment
ast.parse(out)
for path,text in [(HERE/(name+'.py'),out),(HERE/(name+'_fragment.py'),fragment)]:
    with path.open('x') as f:f.write(text)
manifest=dict(base=str(base),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),ranks=ranks,
              source_sha256=hashlib.sha256(out.encode()).hexdigest(),method='G[:,I] inv(G[I,I]), row factors and sampled Gram core')
with (HERE/(name+'_manifest.json')).open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps(manifest))
