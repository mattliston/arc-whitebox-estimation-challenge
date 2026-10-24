from pathlib import Path
import importlib.util,json,time
import torch as t
t.set_num_threads(1);t.set_num_interop_threads(1)
P=Path(__file__).resolve().parent
mods=[]
for name in ('torch_rtt_current95c_v3','torch_push_current95c_rtt_v2'):
 s=importlib.util.spec_from_file_location(name,P/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);mods.append(m)
port,push=mods;t.manual_seed(827);start=time.monotonic();out=[]
for truncated,mapping in ((False,False),(True,False),(True,True)):
 n,k,q=48,24,6;old=16 if mapping else 8
 specs=((0,8,((0,4),)),(8,16,((0,4),))) if mapping else ((0,8,((0,4),)),)
 args=(t.randn(n,k),t.randn(k),t.randn(n,q),t.arange(32),t.arange(40),t.randn(32,dtype=t.float64),t.randn(old,dtype=t.float64),t.randn(old,dtype=t.float64),t.arange(old-1,-1,-1))
 args+=tuple(t.randn(q,hi-lo) for lo,hi,pairs in specs)
 if mapping:args+=(t.randn(q,q),)
 def leaves():return tuple(x.clone().requires_grad_(x.is_floating_point()) for x in args)
 a,b=leaves(),leaves();static=(specs,old,truncated,3,mapping)
 direct=port.old_project(t,push,static,*a);checked=port.old_project_checkpoint(t,push,static,b)
 assert t.equal(direct,checked)
 direct.square().mean().backward();checked.square().mean().backward()
 for x,y in zip(a,b):
  if x.is_floating_point():assert t.equal(x.grad,y.grad) and t.isfinite(x.grad).all()
 out.append(dict(truncated=truncated,mapping=mapping,forward_exact=True,all_gradient_bytes_exact=True))
r=dict(status='PURE_INNER_CHECKPOINT_FORWARD_AND_GRADIENT_PASS',cases=out,seconds=time.monotonic()-start)
with (P/'TINY_INNER_CHECKPOINT_v1.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
