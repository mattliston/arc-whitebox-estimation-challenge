"""CPU-only safe tensor extraction and reversible covariance-source insertion."""
import ast
import numpy as np
import torch
from common import HERE,A,PARENT,MODEL,read,write,pin,seal_check
seal_check();torch.set_num_threads(1)
saved=torch.load(MODEL,map_location='cpu',weights_only=True)
assert saved['use_joint'] is True
shapes=dict(left=(14,16),right=(14,16),edge=(3,16),bias=(16,),output=(16,))
state=saved['state_dict'];assert set(state)==set(shapes)
p={}
for k,v in state.items():
    assert not v.requires_grad and tuple(v.shape)==shapes[k] and v.device.type=='cpu' and v.dtype==torch.float32
    p[k]=v.detach().numpy().copy()
p['feature_scale']=saved['feature_scale'].detach().numpy().copy()
for k in ['rho_scale','third_scale']:p[k]=np.asarray(saved[k],np.float32)
p['cov_factor']=np.asarray(saved['cov_scale']/(saved['sigma_scale2']*saved['envelope_scale']),np.float32)
assert p['feature_scale'].shape==(14,) and (p['feature_scale']>0).all()
assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in p.values())
assert all(float(p[k])>0 for k in ['rho_scale','third_scale','cov_factor'])
literal={k:v.tolist() for k,v in p.items()}
assert all(np.asarray(literal[k],np.float32).tobytes()==v.tobytes() for k,v in p.items())
source=PARENT.read_text();lines=source.splitlines(keepends=True)
n=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'][-1]
original=''.join(lines[n.lineno-1:n.end_lineno])
old='            if hook is not None:\n                dm = fnp.asarray(hook(l, m, C, K21, aux, W[l]), fnp.float64)'
new='            if _EDGE_ENABLED and 8 <= l <= 14:\n                C = C + _edge_source(l,m,K21,aux)\n'+old
assert original.count(old)==1
changed=original.replace(old,new,1)
prefix=''.join(lines[:n.lineno-1])+changed+''.join(lines[n.end_lineno:])
assert prefix.replace(changed,original,1)==source
suffix='\n\n_EDGE_LITERAL='+repr(literal)+'\n'+(HERE/'edge_helper.py').read_text()
candidate=prefix+suffix;ast.parse(candidate)
assert candidate[:-len(suffix)].replace(changed,original,1)==source
with (HERE/'candidate.py').open('x') as f:f.write(candidate)
np.savez(HERE/'parameters.npz',**p)
oldverdict=read(A/'joint_edge_covariance_verdict_v1.json')
assert not oldverdict['information_gate_pass']
write(HERE/'build.json',dict(candidate=pin(HERE/'candidate.py'),parameters=pin(HERE/'parameters.npz'),
    original_model=pin(MODEL),parent=pin(PARENT),inverse_byte_exact=True,parameter_scalars=sum(v.size for v in p.values()),
    original_information_gate_pass=False,new_fit_steps=0,truth_accessed=False,CPU_only=True,torch_version=torch.__version__))
print(dict(candidate=pin(HERE/'candidate.py'),parameter_scalars=sum(v.size for v in p.values()),inverse_byte_exact=True),flush=True)
