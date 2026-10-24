"""Fixed-D implementation-only requalification, using saved native outputs.

Run in the native Python runtime; native_bridge appends the compatible Torch
site after loading native NumPy/flops. No target, new direction, amplitude,
model fit, or native final-block rerun occurs here.
"""
import native_bridge as bridge
import math
import numpy as np
import torch
from common import HERE,read,pin,once,packet
from consumer_v2 import tensors,anchored,star_plane,star_factor
from model import Pair,inputs,project

torch.set_num_threads(1)
old=read(HERE/'qualification.json');assert not old['passed']
assert old['direction']==pin(HERE/'qualification_direction.npz')
assert old['predictions']==pin(HERE/'qualification_predictions.npz')
with np.load(packet(38),allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
with np.load(HERE/'parent_parameters.npz',allow_pickle=False) as z:P=tensors({k:z[k].copy() for k in z.files})
with np.load(old['direction']['path'],allow_pickle=False) as z:
    D=torch.from_numpy(z['D'].copy())
    norm={k[len('normalizer_'):]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('normalizer_')}
with np.load(old['predictions']['path'],allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
a=tensors(data);n=len(a['v']);assert n==1024

def selected(v):
    z=np.abs(data['mu']/np.sqrt(np.maximum(v,1e-10)))
    count=min(n,((int(np.count_nonzero(z<=3.5))+31)//32)*32)
    return np.argsort(z,kind='stable')[:count]

predictions={};variances={};columns={};rowsets={};exact_variance={}
with torch.no_grad():
    for sign in (0,1,-1):
        key=str(sign)
        y=anchored(a,P,sign*D)
        predictions[key]=y.numpy().copy()
        v=a['_last_physical_v'].detach().numpy().copy()
        variances[key]=v
        exact_variance[key]=v.tobytes()==saved[f'variance_{key}'].tobytes()
        assert exact_variance[key],('native variance bytes',sign)
        rowsets[key]=selected(v)
        columns[key]=star_factor(star_plane(a,a['C'].double()+sign*D),True)[2].tolist()
        assert np.isfinite(y.numpy()).all() and np.min(v)>1e-10

# Exactly the same synthetic head-coordinate fixture and fresh trunk as v1.
# No model choice or new direction is introduced by the implementation fix.
torch.manual_seed(2026092766)
model=Pair()
with torch.no_grad():
    model.output.weight.copy_(torch.sin(torch.arange(1,33,dtype=torch.float32))[None,:]/math.sqrt(32))
    model.output.bias.zero_()
Dtrain=project(model(*inputs(a,norm)),a['m14'].double())
assert torch.equal(Dtrain.detach(),D),'fixed direction must remain byte-identical'
ytrain=anchored(a,P,Dtrain)
loss=(ytrain-a['prediction'][-1]).square().mean()*1e8
grads=torch.autograd.grad(loss,tuple(model.parameters()))
assert all(torch.isfinite(g).all() for g in grads)
grad_norm=float(torch.sqrt(sum(g.double().square().sum() for g in grads)))
assert math.isfinite(grad_norm) and grad_norm>0

comparisons=[]
for sign in (1,-1):
    key=str(sign)
    native=saved[f'native_{key}']-saved['native_0']
    proposed=predictions[key]-predictions['0']
    nr=float(np.linalg.norm(native));pr=float(np.linalg.norm(proposed))
    error=float(np.linalg.norm(proposed-native))
    rel=error/max(nr,1e-300)
    cosine=float(np.dot(proposed,native)/max(nr*pr,1e-300))
    rms=nr/math.sqrt(n)
    comparisons.append(dict(sign=sign,native_response_rms=rms,relative_l2=rel,cosine=cosine,
        absolute_error_rms=error/math.sqrt(n),passed=bool(rms>1e-9 and rel<=.01 and cosine>=.999)))
zero=np.array_equal(predictions['0'],data['prediction'][-1])
native_membership=all(np.array_equal(np.sort(saved['rows_0']),np.sort(saved[f'rows_{sign}'])) for sign in (1,-1))
torch_native_membership=all(np.array_equal(np.sort(rowsets[str(sign)]),np.sort(saved[f'rows_{sign}'])) for sign in (0,1,-1))
columns_stable=columns['0']==columns['1']==columns['-1']
passed=zero and native_membership and torch_native_membership and columns_stable and all(x['passed'] for x in comparisons)
outpath=HERE/'qualification_v2_predictions.npz'
with outpath.open('xb') as f:
    np.savez(f,**{f'hybrid_{k}':v for k,v in predictions.items()},**{f'variance_{k}':v for k,v in variances.items()})
once(HERE/'qualification_v2.json',dict(passed=passed,rank=38,exact_zero=zero,
    native_variance_forward_bytes_exact=exact_variance,native_row_membership_stable=native_membership,
    torch_native_row_membership_equal=torch_native_membership,torch_star_selected_columns_stable=columns_stable,
    comparisons=comparisons,full_width_backward=True,finite_all_parameter_gradients=True,
    backward_gradient_norm=grad_norm,derivative='Physical real-arithmetic variance VJP; straight-through rounding derivative.',
    bridge=bridge.stats(),direction_unchanged=True,direction=old['direction'],
    saved_native_predictions=old['predictions'],original_failed_qualification=pin(HERE/'qualification.json'),
    predictions=pin(outpath),targets_read=False,fit=False,native_final_block_rerun=False,
    sources=[pin(HERE/name) for name in ('native_bridge.py','consumer_v2.py','qualify_v2.py','model.py')]))
print(read(HERE/'qualification_v2.json'),flush=True)
if not passed:raise SystemExit(2)
