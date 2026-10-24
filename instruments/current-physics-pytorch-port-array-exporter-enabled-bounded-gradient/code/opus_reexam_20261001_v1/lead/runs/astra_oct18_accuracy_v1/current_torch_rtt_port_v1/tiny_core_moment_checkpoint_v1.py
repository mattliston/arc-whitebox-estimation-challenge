from pathlib import Path
import importlib.util,json,hashlib,types
import torch as t
from torch.utils.checkpoint import checkpoint
P=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1')
t.set_num_threads(1);t.set_num_interop_threads(1)
sp=importlib.util.spec_from_file_location('moments',P/'core_moment_checkpoint_v1.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
cd=types.SimpleNamespace();m.install(cd,t)
rows=[]
for dtype in (t.float32,t.float64):
 t.manual_seed(1827)
 inputs=[t.randn(33,257,dtype=dtype)*.03,t.randn(257,dtype=dtype)*.02,t.randn(33,dtype=t.float64)*.1,t.randn(33,dtype=t.float64)*.1]
 def block(pp,lam,aa0,beta0,ckpt):
  if ckpt:diag=cd._o18_moment_diag(pp,lam)
  else:
   p64=pp.to(t.float64);diag=((p64*p64*p64)*lam.to(t.float64)).sum(1)
  aa=aa0+diag*.1;beta=beta0-diag*.2
  if ckpt:energy,column=cd._o18_moment_columns(pp,aa,beta)
  else:energy,column=(p64*p64).T@aa,p64.T@beta
  return diag,energy,column
 def run(use):
  x=[a.clone().requires_grad_() for a in inputs]
  if use:out=checkpoint(lambda *z:block(*z,True),*x,use_reentrant=False)
  else:out=block(*x,False)
  loss=sum((a*a).sum() for a in out);loss.backward()
  return tuple(a.detach() for a in out),[a.grad for a in x]
 a,ga=run(False);b,gb=run(True)
 assert all(t.equal(x,y) for x,y in zip(a,b))
 diff=[float((x-y).abs().max()) for x,y in zip(ga,gb)]
 rel=[float((x-y).norm()/x.norm().clamp_min(1e-30)) for x,y in zip(ga,gb)]
 assert all(t.isfinite(g).all() for g in gb)
 assert all(t.allclose(x,y,atol=1e-12,rtol=1e-6) for x,y in zip(ga,gb)),(diff,rel)
 with t.no_grad():c=block(*inputs,True)
 assert all(t.equal(x,y) for x,y in zip(a,c))
 rows.append(dict(dtype=str(dtype),forward_byte_equal=True,gradient_max_abs=diff,gradient_rel_l2=rel,finite=True,nested_checkpoint=True))
receipt=dict(passed=True,rows=rows,helper_sha256=hashlib.sha256((P/'core_moment_checkpoint_v1.py').read_bytes()).hexdigest(),note='Separate cast nodes can differ in float32 gradient accumulation roundoff only; no forward difference.')
with (P/'TINY_CORE_MOMENTS_v1.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
