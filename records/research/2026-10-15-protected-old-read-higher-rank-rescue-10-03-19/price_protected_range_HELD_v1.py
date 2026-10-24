from pathlib import Path
import ast,hashlib,json,importlib.util
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;O=D.parents[2]
C=O/'lead/runs/astra_read_column_tail_rank2_v1/run_capture_train16_v1/native_16_capture.json'
cert=json.loads(C.read_text());shapes=[x for x in cert['capture_metadata']if x['old_columns']]
qr={}
for r in (16,32):
 for a in shapes:
  m=a['m'];height=((3*m+255)//256)*64
  for mode in ('original_center','aligned_height'):
   center=height-(2 if mode=='original_center' else r);tail=m-center
   if (tail,r)in qr:continue
   with fs.BudgetContext(flop_budget=2**50,quiet=True):x=fnp.asarray(np.random.default_rng(tail+r).normal(size=(tail,r)).astype('float32'))
   with fs.BudgetContext(flop_budget=2**50,quiet=True)as ctx:q,rr=fnp.linalg.qr(x,mode='reduced')
   qr[tail,r]=dict(flops=int(ctx.flops_used),ops=len(ctx._op_log))
   del x,q,rr

spec=importlib.util.spec_from_file_location('vt',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest();M,_=p.load()
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32);M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None];M._bind_push6(arena)
M._WINOGRAD=True;M._AB_ON=True;M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
# Reuse the exact tested pricing API; do not execute its old experiment.
tree=ast.parse((D/'price_range32_HELD_v1.py').read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='inner')
exec(compile(ast.Module(body=[node],type_ignores=[]),'reuse-inner-pricer','exec'))

def overhead(m,old,dim,r,center):
 tail=m-center;packed=center+r
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  L=fnp.empty((tail,old),dtype=fnp.float32);omega=fnp.empty((old,r),dtype=fnp.float32);Q=fnp.empty((tail,r),dtype=fnp.float32);I=fnp.empty((r,dim),dtype=fnp.float32)
  central=fnp.empty((center,old),dtype=fnp.float32);compressed=fnp.empty((r,old),dtype=fnp.float32);central_i=fnp.empty((center,dim),dtype=fnp.float32);tail_i=fnp.empty((tail,dim),dtype=fnp.float32)
 parts={}
 for name,fn in [('range',lambda:L@omega),('project',lambda:Q.T@L),('pack',lambda:fnp.concatenate((central,compressed),axis=0)),('lift',lambda:Q@I),('restore',lambda:fnp.concatenate((central_i,tail_i),axis=0))]:
  with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:z=fn()
  parts[name]=dict(flops=int(ctx.flops_used),ops=len(ctx._op_log))
 parts['qr']=qr[tail,r]
 return parts

base={a['layer']:inner(a['layer'],a['m'],a['old_columns'],a['old_rank'])for a in shapes}
rows=[]
for r in (16,32):
 for mode in ('original_center','aligned_height'):
  for a in shapes:
   l,m,old,dim=[a[k]for k in ('layer','m','old_columns','old_rank')];height=((3*m+255)//256)*64;center=height-(2 if mode=='original_center' else r);packed=center+r
   reduced=inner(l,packed,old,dim);extra=overhead(m,old,dim,r,center);debit=sum(x['flops']for x in extra.values())
   row=dict(layer=l,rank=r,mode=mode,m=m,center=center,outer=m-center,packed=packed,old_columns=old,dim=dim,baseline=base[l],reduced=reduced,overhead=extra,warm_saving=base[l]['warm']['flops']-reduced['warm']['flops']-debit,cold_saving=base[l]['cold']['flops']-reduced['cold']['flops']-debit)
   rows.append(row)
summaries=[]
for r in (16,32):
 for mode in ('original_center','aligned_height'):
  subset=[x for x in rows if x['rank']==r and x['mode']==mode];warm=sum(x['warm_saving']for x in subset);cold=sum(x['cold_saving']for x in subset)
  formation=2*r*(6144*384+11712*320+11712*320+4480*384+320*384)
  summaries.append(dict(rank=r,mode=mode,warm_saving_flops=warm,warm_saving_pct=100*warm/cert['flops'],cold_saving_pct=100*cold/cert['flops'],metric_alignment_formation_flops=formation,after_alignment_warm_saving_pct=100*(warm-formation)/cert['flops']))
result=dict(status='SOURCE_ONLY_PROTECTED_OUTER_RANGE_TARIFF',source_sha256=p.SOURCE_SHA,whole_bill=cert['flops'],summaries=summaries,rows=rows,models=0,numerical_payload_bytes=p.PAYLOAD[0],limits=['Ranks16/32 requested prospectively, both original-center and fixed aligned-height cases; no outcome selection.','Outer-only range/project/QR/lift plus both concatenations are charged. Original u/v kernel and right-cache cold/warm tariff measured.','Actual small QR only; large operands virtual with public FNP charging.','Metric alignment is a separately shown conservative dense frame-level formation; random generation/finite guards/hosted requests are not priced.','Aligned-height case removes14/30 additional exact rows and needs its own fidelity assessment.','No accuracy or native-runtime claim.'])
out=D/'PROTECTED_RANGE_TARIFF_v1.json';out.open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summaries=summaries,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),models=0)))
