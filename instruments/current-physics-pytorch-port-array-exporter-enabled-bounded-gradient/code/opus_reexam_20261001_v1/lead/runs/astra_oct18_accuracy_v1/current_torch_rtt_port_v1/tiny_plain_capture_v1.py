"""Synthetic exact M-style replay graph: differentiable plain then constant carrier."""
import importlib.util,json,types,time
from pathlib import Path
import torch as t
from torch.utils.checkpoint import checkpoint
t.set_num_threads(1); t.set_num_interop_threads(1)
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('push',P/'torch_push_current95c_rtt_v2.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
t.manual_seed(713);start=time.monotonic()
x=t.randn(24,8,requires_grad=True);z=t.randn(8,6,requires_grad=True)
cd=types.SimpleNamespace(call_layer=14);state={'again':False};forward={}
def layer(x,z):
    again=state['again'];state['again']=True;cd._o18_rtt_base_old=None
    def plain():
        old=x@z;lam=z.sum(1).square()
        mod.capture_rtt_old(cd,t,old,lam)
        return old.square()
    p=plain();saved_capture=cd._o18_rtt_base_old
    if not again:
        with t.no_grad(): forward['full']=plain()+.01
        assert cd._o18_rtt_base_old is saved_capture
    value=forward['full']+(p-p.detach())
    _,old,lam=cd._o18_rtt_base_old
    assert old.requires_grad and lam.requires_grad
    result=value-old+(x.square()*lam[None,:])@z
    cd._o18_rtt_base_old=None
    if not again:forward['out']=result.detach().clone()
    else:assert t.equal(forward['out'],result.detach())
    return result
out=checkpoint(layer,x,z,use_reentrant=False)
out.square().mean().backward()
assert all(a.grad is not None and t.isfinite(a.grad).all() and a.grad.abs().sum()>0 for a in (x,z))
r=dict(status='M_STYLE_NONREENTRANT_CHECKPOINT_METADATA_AND_VALUE_PASS',seconds=time.monotonic()-start,gradient_norms=[float(a.grad.norm()) for a in (x,z)],no_checks_disabled=True)
with (P/'TINY_PLAIN_CAPTURE_v1.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
