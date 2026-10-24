"""Source inverse, mock-reference identities, and import only; no model calls."""
from pathlib import Path
import ast, hashlib, json, sys, types
import numpy as np

D=Path(__file__).resolve().parent
O=D.parents[2]
parent=O/'lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py'
fragment=D/'fragment_intermediate_capture_HELD_v1.py'
candidate=D/'candidate_QR_intermediate_capture_HELD_v1.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert candidate.read_bytes()==parent.read_bytes()+fragment.read_bytes()
tree=ast.parse(fragment.read_text())
assert not any(isinstance(n,ast.Name) and n.id in ('fnp','flops','np') for n in ast.walk(tree))
assert not any(isinstance(n,ast.Attribute) and n.attr in ('copy','copyto','astype','matmul') for n in ast.walk(tree))

arrays={l:dict(host=np.full(2,float(l)),x=np.full((2,30),l,dtype=np.float32),
              sigma=np.ones(2),delta=np.full(2,.1,dtype=np.float32),
              state=np.full((2,64),l,dtype=np.float32),effective=np.full(2,.1)) for l in range(7,15)}
cache={2:dict(layers={l:dict(head=np.zeros(64),head_b=np.array(0.),rms=np.array(1.),gain=np.array(1.)) for l in range(7,15)})}
calls=dict(features=0,gnn=0,fhm=0,predict=0)
marker=object()
def fake_features(l,m,C,K21,aux,W):
    calls['features']+=1
    return arrays[l]['x']
def fake_gnn(l,x,sigma,W,state,P):
    calls['gnn']+=1
    return arrays[l]['delta'],arrays[l]['state']
def fake_fhm(l,mean,delta):
    calls['fhm']+=1
    return arrays[l]['effective']
ns=dict(features=fake_features,gnn_step=fake_gnn,_astra_fhm_effective=fake_fhm,_E2_GNN_STATE=cache)
class MockEstimator:
    def predict(self, fail=False):
        calls['predict']+=1
        for l,a in arrays.items():
            x=ns['features'](l,a['host'],None,None,None,None)
            delta,state=ns['gnn_step'](l,x,a['sigma'],None,None,None)
            effective=ns['_astra_fhm_effective'](l,a['host'],delta)
            assert x is a['x'] and state is a['state'] and delta is a['delta'] and effective is a['effective']
        if fail:raise RuntimeError('mock exception')
        return marker
ns['Estimator']=MockEstimator
exec(compile(fragment.read_text(),str(fragment),'exec'),ns)
assert MockEstimator().predict() is marker
assert ns['_NLR_ACTIVE'] is False and set(ns['_NLR_CAPTURE'])==set(range(7,15))
for l,a in arrays.items():
    for k,target in [('host','host'),('features30','x'),('sigma','sigma'),('delta','delta'),('state64','state'),('effective_delta','effective')]:
        assert ns['_NLR_CAPTURE'][l][k] is a[target]
assert calls==dict(features=8,gnn=8,fhm=8,predict=1)
try:MockEstimator().predict(fail=True)
except RuntimeError as e:assert str(e)=='mock exception'
else:raise AssertionError('mock exception missing')
assert ns['_NLR_ACTIVE'] is False

# Full source import, no Estimator construction/setup/predict.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
m=types.ModuleType('_nlr_import');m.__file__=str(candidate);sys.modules[m.__name__]=m
exec(compile(candidate.read_text(),str(candidate),'exec',dont_inherit=True),m.__dict__)
assert m._NLR_ACTIVE is False and m._NLR_CAPTURE=={}
assert m._NLR_PARENT_GNN is not m.gnn_step and m._NLR_PARENT_FHM is not m._astra_fhm_effective
assert m._RTP_ON is True
for p in (D/'native_intermediate_capture_HELD_v1.py',D/'finite_intermediate_capture_HELD_v1.py',D/'accept_capture_retirement_v1.py'):
    ast.parse(p.read_text())
out=dict(status='PASS_SOURCE_INVERSE_ZERO_FNP_MOCK_REFERENCE_IMPORT_ONLY',parent_sha256=sha(parent),candidate_sha256=sha(candidate),fragment_sha256=sha(fragment),source_inverse=True,observer_FNP_operations=0,all_mock_return_and_capture_identities=True,exception_cleanup=True,real_model_calls=0,source_file_sha256=sha(__file__),retained_payload_estimate_bytes=8*1024*(30*4+64*4+6*8),deployment_planning_flops=8*(2*1024*94+4*1024),deployment_cost_not_native_measured=True)
p=D/'SOURCE_TINY_READY_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out));print('receipt_sha256',sha(p))
