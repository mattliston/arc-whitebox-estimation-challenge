"""Retain parent's complete mean correction; keep lower-risk vector identities."""
import ast, hashlib, json
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent/'opus_fix336687_v1/candidate_v61j.py'
parent=P.read_text()
assert hashlib.sha256(parent.encode()).hexdigest()=='ac93b10f2c8fa541b94cf0787b7f573d36cf1233df219d00a22335c4f9a47129'
fragment=(D/'fragment_bookkeeping_algebra_v1.py').read_text()
a=parent.index('                _gn_C0 = C\n',parent.index('def _astra_fhm_core'))
b=parent.index('                _ps_post_cov32 = C\n',a)
c=fragment.index('                # Real-arithmetic identity')
d=fragment.index('                _ps_post_cov32 = C\n',c)
fragment=fragment[:c]+parent[a:b]+fragment[d:]
fragment=fragment.replace('# ASTRA bookkeeping v1: algebra','# ASTRA bookkeeping v2: parent mean correction retained',1)
tree=ast.parse(parent)
node=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='_bd_update'][-1]
bd='\n'.join(parent.splitlines()[node.lineno-1:node.end_lineno])+'\n'
old='fnp.asarray(_S19_BD_ADD[:3], dtype=fnp.float64)'
assert bd.count(old)==1
bd=bd.replace(old,'_g5_arr(_S19_BD_ADD[:3])',1)
bdfrag='\n# Immutable literal F64 triple, cached read-only by the existing G5 owner.\n'+bd+'''\n_ASTRA_DENSE_CONTEXT = tuple((name, _bd_update if name == '_bd_update' else value)
                             for name, value in _ASTRA_DENSE_CONTEXT)
'''
fragment+=bdfrag
candidate=parent+'\n'+fragment
compile(candidate,'candidate_v61j_bookkeeping_v2','exec')
F=D/'fragment_bookkeeping_algebra_v2.py'
Q=D/'candidate_v61j_bookkeeping_algebra_HELD_v2.py'
F.open('x').write(fragment)
Q.open('x').write(candidate)
receipt=dict(parent=str(P),parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),candidate=str(Q),sha256=hashlib.sha256(candidate.encode()).hexdigest(),fragment=str(F),restored='Parent entire FP32 covariance mean update and original F64 mean-correction diagonal, including copied diagonal ownership.',remaining='2*bpre cast commutation; cached gq cast; repeated(3*u)*u reuse; four float64 diagonal identities; immutable S19_BD_ADD triple cache.',expected_requests_saved='~223 cold/~224 warm, estimate from v1 native338 minus mean129 plus literalcache14/15.',prediction='No changed matrix FP32 arithmetic; float64 diagonal reassociation can still perturb rounding. No full model measured.')
(D/'BOOKKEEPING_BUILD_v2.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
