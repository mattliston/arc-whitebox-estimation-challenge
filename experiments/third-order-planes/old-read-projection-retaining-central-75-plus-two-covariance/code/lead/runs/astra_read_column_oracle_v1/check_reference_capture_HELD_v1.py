from pathlib import Path
import hashlib,json
import numpy as np
D=Path(__file__).resolve().parent
s=(D/'fragment_reference_capture_HELD_v1.py').read_text()
ns={'np':np,'calls':[],'_RTT_STATS':{'reads':[]},'_E2_LAYER':{'ecut_rows':np.array([0,1]),'mu':np.array([1.,2.]),'vc':np.array([3.,4.]),'g':np.array([.1,.2])}}
exec('''
def refine(prefix,Q,inner,old_right,hold):
 calls.append('refine');return Q,2,False
def read(left,right,lam,cols,state):
 calls.append('read')
 Q=np.eye(2);answer=_rtp_refine(left,Q,None,None,None)
 _RTT_STATS['reads'].append(dict(layer=2,changed=True,c15=1,c2=2))
 return left
_rtt_read=read
_RC_PARENT_READ=read
_rtp_refine=refine
''',ns)
exec(compile(s,'fragment','exec'),ns)
a=np.arange(6,dtype=np.float32).reshape(2,3);b=a.copy();lam=np.ones(3);cols=np.array([0,1]);before=[x.tobytes() for x in (a,b,lam,cols)]
seen=[]
ns['_RCO_OBSERVER']=lambda r,p:seen.append((r,p))
out=ns['_RC_PARENT_READ'](a,b,lam,cols,{'columns':0})
assert out is a and ns['calls']==['read','refine'] and len(seen)==1
assert seen[0][1]['left'] is a and seen[0][1]['right'] is b
assert seen[0][1]['refine']['prefix'] is a
assert before==[x.tobytes() for x in (a,b,lam,cols)] and ns['_RCO_CONTEXT']==[None]
ns['_RCO_OBSERVER']=None
out=ns['_RC_PARENT_READ'](a,b,lam,cols,{'columns':0});assert out is a and len(seen)==1
ns['_RCO_OBSERVER']=lambda r,p:(_ for _ in ()).throw(RuntimeError('observer deliberate failure'))
try:ns['_RC_PARENT_READ'](a,b,lam,cols,{'columns':0})
except RuntimeError:pass
else:raise AssertionError('exception missing')
assert ns['_RCO_CONTEXT']==[None]
r=dict(passed=True,scope='tiny delegates/reference identity/input bytes/observer exception cleanup; no model or full candidate import',suffix_sha256=hashlib.sha256(s.encode()).hexdigest(),new_fnp_calls=0)
p=D/'REFERENCE_CAPTURE_TINY_v1.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(p.name,hashlib.sha256(p.read_bytes()).hexdigest())
