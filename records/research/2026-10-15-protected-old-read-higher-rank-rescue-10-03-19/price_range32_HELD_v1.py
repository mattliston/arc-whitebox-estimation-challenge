from pathlib import Path
import ast,hashlib,json,importlib.util
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;O=D.parents[2]
C=O/'lead/runs/astra_read_column_tail_rank2_v1/run_capture_train16_v1/native_16_capture.json'
cert=json.loads(C.read_text());shapes=[x for x in cert['capture_metadata']if x['old_columns']]
# Actual small QR tariff, before the metadata-only matrix backend is installed.
qr={}
for m in sorted({a['m']for a in shapes}):
 with fs.BudgetContext(flop_budget=2**50,quiet=True):a=fnp.asarray(np.random.default_rng(m).normal(size=(m,32)).astype('float32'))
 with fs.BudgetContext(flop_budget=2**50,quiet=True)as ctx:q,r=fnp.linalg.qr(a,mode='reduced')
 qr[m]=dict(flops=int(ctx.flops_used),ops=len(ctx._op_log))
 del a,q,r

spec=importlib.util.spec_from_file_location('vt',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest();M,_=p.load()
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32);M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None];M._bind_push6(arena)
M._WINOGRAD=True;M._AB_ON=True;M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False

def inner(layer,m,old,r):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  left=fnp.empty((m,old),dtype=fnp.float32);lam=fnp.empty((old,),dtype=fnp.float32)
  if layer<10:F=fnp.empty((old,r),dtype=fnp.float32)
  else:
   P=5856;uv=(fnp.empty((256,P),dtype=fnp.float32),fnp.empty((320,P),dtype=fnp.float32),[(0,P)],256)
   M._UBC_RUN[0]=dict(uv=uv,owner=uv[0],generation=('range32-price',),cache={},bytes=0)
   if layer>=13:
    J=(old-11712)//2;uvj=(fnp.empty((256,J),dtype=fnp.float32),fnp.empty((384,J),dtype=fnp.float32),[(0,J)],256);mp=fnp.empty((320,384),dtype=fnp.float32)
 def call():
  if layer<10:return M._shared_old384_mm(left*lam[None,:],F)
  z=M._uvp_inner(left[:,:11712],lam[:11712],uv)
  if layer>=13:return M._shared_old384_mm(z,mp)+M._uvp_inner(left[:,11712:],lam[11712:],uvj)
  return z
 result={}
 for phase in ('cold','warm'):
  with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:z=call()
  result[phase]=dict(flops=int(ctx.flops_used),ops=len(ctx._op_log),shape=list(z.shape))
 M._BH_HOLD.clear();M._UBX_STATE['cache'].clear();M._UBX_STATE['bytes']=0;M._UBX_STATE['run']=None;M._UBC_RUN[0]=None
 return result

def thin(m,old,r):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  L=fnp.empty((m,old),dtype=fnp.float32);omega=fnp.empty((old,32),dtype=fnp.float32);Q=fnp.empty((m,32),dtype=fnp.float32);I=fnp.empty((32,r),dtype=fnp.float32)
 parts={}
 for name,fn in [('range',lambda:L@omega),('project',lambda:Q.T@L),('lift',lambda:Q@I)]:
  with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:z=fn()
  parts[name]=dict(flops=int(ctx.flops_used),ops=len(ctx._op_log))
 return parts

rows=[]
for a in shapes:
 l,m,old,r=[a[k]for k in ('layer','m','old_columns','old_rank')]
 base=inner(l,m,old,r);small=inner(l,32,old,r);extra=thin(m,old,r)
 overhead=sum(v['flops']for v in extra.values())+qr[m]['flops']
 rows.append(dict(layer=l,m=m,old_columns=old,rank=r,baseline=base,reduced=small,thin=extra,qr=qr[m],warm_saving=base['warm']['flops']-small['warm']['flops']-overhead,cold_saving=base['cold']['flops']-small['cold']['flops']-overhead))
 print(l,rows[-1]['warm_saving'],flush=True)
warm=sum(r['warm_saving']for r in rows);cold=sum(r['cold_saving']for r in rows)
# Aligning sketch with H=L*diag(lambda)*F_eff.T costs one extra frame formation.
# Materialized L7/L10 factors; L13 uses base+map and recent factors. Dense upper bound
# shown as separate accounting; no claim raw-L sketch attains H's spectral oracle.
formation=2*32*(6144*384+11712*320+11712*320+4480*384+320*384)
result=dict(status='SOURCE_ONLY_ACTUAL_FNP_TARIFF',source_sha256=p.SOURCE_SHA,rank=32,rows=rows,warm_saving_flops=warm,warm_saving_pct=100*warm/cert['flops'],cold_saving_flops=cold,cold_saving_pct=100*cold/cert['flops'],sketch_metric_alignment_dense_formation_flops=formation,after_alignment_warm_saving_pct=100*(warm-formation)/cert['flops'],whole_bill=cert['flops'],numerical_payload_bytes=p.PAYLOAD[0],model_calls=0,limits=['One fixed rank32, all old reads L7..14; same full bank/reconstruction/outer products.','Actual public FNP charging and existing kernels; matrix values are virtual. QR is actual small FNP.','Warm right trees use same original coefficient owner. Per-shape cold figures are conservative separate starts, not an exact whole-network cold profile.','Dense random sketch generation, orthogonalization stability fallback, finite checks, extra requests and hosted setup are not yet included.','Raw-L randomized range is not an old-inner oracle; alignment requires the separately stated frame formation.','No accuracy claim; saved oracle is owned by bigger_gain_hunt.'])
(D/'RANGE32_TARIFF_v1.json').open('x').write(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('rows','limits')}))
