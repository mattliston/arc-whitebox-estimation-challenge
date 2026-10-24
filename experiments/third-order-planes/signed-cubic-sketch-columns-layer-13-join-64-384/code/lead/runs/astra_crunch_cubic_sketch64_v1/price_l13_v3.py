"""Current public weighted tariff of final-join-only hybrid64; metadata backend."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
s=importlib.util.spec_from_file_location('virtual_tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py';p.SOURCE_SHA='46e9315d1a3e563cf669f27ec499be487ca7a437ee942b7be44ea1fad682ad9f'
old_backend=p.backend
def backend(fn,*args,**kw):
 if getattr(fn,'__name__','')=='sum':
  a=args[0];axis=kw.get('axis',args[1] if len(args)>1 else None);shape=() if axis is None else tuple(v for i,v in enumerate(a.shape) if i!=axis%a.ndim)
  return p.virtual(shape,kw.get('dtype') or a.dtype)
 return old_backend(fn,*args,**kw)
p.backend=backend;M,_=p.load();f=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 arena=f.empty((M._PACKED_ELEMENTS,),dtype=f.float32)
 M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None];M._bind_push6(arena)
M._WINOGRAD=True;M._AB_ON=True;M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
def meter(fn):
 with fs.BudgetContext(flop_budget=2**60,quiet=True) as ctx:out=fn()
 return out,dict(flops=int(ctx.flops_used),charged_ops=len(ctx._op_log),operations=ctx.summary_dict()['operations'])
rows=[]
for m,k in ((800,4800),(832,4544)):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  J=f.empty((m,k),dtype=f.float32);coef=f.empty((k,),dtype=f.float32);weight=f.empty((k,),dtype=f.float32)
  Qp=f.empty((m,384),dtype=f.float32);core=f.empty((384,384),dtype=f.float32)
  selected={r:f.empty((r,k),dtype=f.float32) for r in (320,384)}
  oldsel={r:f.empty((r,384),dtype=f.float32) for r in (320,384)}
  omega=f.empty((m,64),dtype=f.float32)
 def gram(r):
  Y=M._shared_old384_mm(J,weight[:,None]*selected[r].T)
  return Y+M._shared_old384_mm(Qp,M._shared_old384_mm(core,oldsel[r].T))
 def sketch():
  H=M._shared_old384_mm(omega.T,J).T
  norms=f.sum(J*J,axis=0)/m
  Z=M._shared_old384_mm(J,coef[:,None]*(H*H-norms[:,None]))
  return Z/f.sqrt(f.sum(Z*Z,axis=0))[None,:]
 # Warm each shape once; core/QR/reprojection/reconstruction are common.
 meter(lambda:gram(384));meter(lambda:gram(320));meter(sketch)
 _,parent=meter(lambda:gram(384));Y,retained=meter(lambda:gram(320));Z,cubic=meter(sketch)
 _,packing=meter(lambda:f.concatenate((Y,Z),axis=1))
 new=retained['flops']+cubic['flops']+packing['flops'];delta=new-parent['flops']
 row=dict(m=m,k=k,old_rank=384,parent=parent,retained=retained,cubic=cubic,packing=packing,delta_flops=delta,delta_wholebill_pct=100*delta/244596689874,delta_charged_ops=retained['charged_ops']+cubic['charged_ops']+packing['charged_ops']-parent['charged_ops'])
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('parent','retained','cubic','packing')}),flush=True)
out=dict(source_sha256=p.SOURCE_SHA,rows=rows,models=0,limits=['Current weighted FNP shape tariff; no native wall/request measurements.','Final L13 only. SingleQR remains m x384 and factor/reconstruction work cancels exactly in shape. No futurecore.','Top320 selection versus top384 has same full sort; reduced gathers and probe-literal row gathers omitted.','Probe normalization/centering, extra packing and old Gram products included.','Representative saved651 and saved7103248 last-join geometries, current oldrank384 substituted. Warm caches.'])
(D/'TARIFF_v3.json').open('x').write(json.dumps(out,indent=2)+'\n')
