from pathlib import Path
import ast,importlib.util,json,hashlib,os
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
s=importlib.util.spec_from_file_location('virtual_tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py';p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest();M,_=p.load();fnp=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32);M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None];M._bind_push6(arena)
M._YR7_ACTIVE[0]=True;M._WINOGRAD=True;M._AB_ON=True;M._YR7_STATS=dict(calls=0,chunks=0,shapes=[],routed_shapes=[],old_fused=0,max_arena_elements=0);M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
R=O/'lead/runs/astra_read_column_tail_rank2_v1/run_capture_train16_v1';cert=json.loads((R/'native_16_capture.json').read_text());rows=[]
def price_inner(layer,m,old,r):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  left=fnp.empty((m,old),dtype=fnp.float32);lam=fnp.empty((old,),dtype=fnp.float32)
  if layer<10: F=fnp.empty((old,r),dtype=fnp.float32)
  else:
   P=5856;uv=(fnp.empty((256,P),dtype=fnp.float32),fnp.empty((320,P),dtype=fnp.float32),[(0,P)],256)
   M._UBC_RUN[0]=dict(uv=uv,owner=uv[0],generation=('protected-price',),cache={},bytes=0)
   if layer>=13:
    J=(old-11712)//2;uvj=(fnp.empty((256,J),dtype=fnp.float32),fnp.empty((384,J),dtype=fnp.float32),[(0,J)],256);mp=fnp.empty((320,384),dtype=fnp.float32)
 def call():
  if layer<10:return M._shared_old384_mm(left,F)
  z=M._uvp_inner(left[:,:11712],lam[:11712],uv)
  if layer>=13:return M._shared_old384_mm(z,mp)+M._uvp_inner(left[:,11712:],lam[11712:],uvj)
  return z
 with fs.BudgetContext(flop_budget=2**60,quiet=True):call()
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:call()
 M._BH_HOLD.clear();return int(ctx.flops_used)
for a in cert['capture_metadata']:
 if not a['old_columns']:continue
 l,m,old,r=[a[x]for x in ('layer','m','old_columns','old_rank')];new=((3*m+255)//256)*64;center=new-2
 baseline=price_inner(l,m,old,r);reduced=price_inner(l,new,old,r)
 extra=4*m*old+new*old+4*(m-center)*r+m*r+32*m
 rows.append(dict(layer=l,m=m,new_rows=new,old_columns=old,rank=r,baseline_inner=baseline,reduced_inner=reduced,projection_concat_restore=extra,saving=baseline-reduced-extra));print(l,rows[-1]['saving'],flush=True)
saved=sum(x['saving']for x in rows);base=sum(x['baseline_inner']for x in rows)
out=dict(status='SOURCE_ONLY_PROTECTED_OUTER_QUARTER_WARM_TARIFF',rows=rows,inner_total=base,inner_total_pct=100*base/cert['flops'],saving=saved,saving_pct=100*saved/cert['flops'],limits=['Current actualFNP contraction kernels on TRAIN16 shapes, no models.','UVP base B caches warmed for each shape, lazyjoin left uncached as production. L7..9 direct384 inner.','Pointwise pair preparation included. Rank2 projection/concat/restore charged explicitly by float32 operation count.','One fixed center=ceil(.75m/64)*64-2. Basis QR and cold changes not measured; warm planning only.'])
p=D/'PROTECTED_OUTER_QUARTER_TARIFF_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('rows','limits')}));print(hashlib.sha256(p.read_bytes()).hexdigest())
