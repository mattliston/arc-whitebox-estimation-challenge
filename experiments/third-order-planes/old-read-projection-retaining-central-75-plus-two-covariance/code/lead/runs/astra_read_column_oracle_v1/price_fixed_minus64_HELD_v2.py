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
def price(a,c):
 m,k,r,old,c2=[a[t]for t in('m','young_k','old_rank','old_columns','c2')];t=c2-c
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  left=fnp.empty((m,old+k),dtype=fnp.float32);right=fnp.empty((c,k),dtype=fnp.float32);lam=fnp.empty((old+k,),dtype=fnp.float32);idx=fnp.asarray(p.np.arange(c,dtype='int64'));inner=fnp.empty((m,r),dtype=fnp.float32);oldright=fnp.empty((c,r),dtype=fnp.float32);state=dict(columns=old,pre_basis=fnp.empty((1024,r),dtype=fnp.float32))
  Q=fnp.empty((m,2),dtype=fnp.float32);scores=fnp.empty((2,k),dtype=fnp.float32);bt=fnp.empty((t,k),dtype=fnp.float32);la=fnp.empty((k,),dtype=fnp.float32);Bprefix=fnp.empty((c,r),dtype=fnp.float32);Btail=fnp.empty((t,r),dtype=fnp.float32)
 M._cp_slr_inner_sq=lambda *args:inner;M._xsg_rows=lambda *args:oldright
 M._TR_LAST[0]=None
 with fs.BudgetContext(flop_budget=2**60,quiet=True):M._yr7_read(left,right,lam,idx,state)
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:prefix=M._yr7_read(left,right,lam,idx,state)
 read=int(ctx.flops_used);ops=len(ctx._op_log);M._BH_HOLD.clear();M._TR_LAST[0]=None
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:
  V=prefix.T@Q
  if old:V=V-Bprefix@(inner.T@Q)
  U=prefix@V
  if old:U=U-inner@(Bprefix.T@V)
  weighted=bt*la;tail=Q@(scores@weighted.T)
  if old:tail=tail+inner@Btail.T
  answer=fnp.concatenate((prefix,tail),axis=1)
 # Q initial/refined QR and Q.T A do not change shape. Both old frame gathers
 # total c2*r elements across prefix/tail, so their tariff cancels.
 return dict(flops=read+int(ctx.flops_used),charged_ops=ops+len(ctx._op_log),read_flops=read,shape_dependent_QR_tail_flops=int(ctx.flops_used))

def selection_price(a):
 m,k,c2=[a[t] for t in ('m','young_k','c2')]
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  A=fnp.empty((8,k),dtype=fnp.float32);B=fnp.empty((c2,k),dtype=fnp.float32);Q=fnp.empty((8,2),dtype=fnp.float32);g=fnp.empty((c2,),dtype=fnp.float32)
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:
  Y=A@B.T;res=Y-Q@(Q.T@Y);squared=res*res
 return int(ctx.flops_used)+8*c2+2*c2,len(ctx._op_log)+3
for a in cert['capture_metadata']:
 l=a['layer'];c=a['c15']
 if c>=a['c2']:continue
 q=c-64;before=price(a,c);after=price(a,q);sample,ops=selection_price(a)
 rows.append(dict(layer=l,original=c,selected_count=q,incumbent_flops=before['flops'],new_flops=after['flops'],gross_saved_flops=before['flops']-after['flops'],sample8_proxy_flops=sample,sample_charged_ops=ops))
 print(l,rows[-1]['gross_saved_flops'],sample,flush=True)
saved=sum(x['gross_saved_flops']for x in rows);selection=sum(x['sample8_proxy_flops']for x in rows)
out=dict(status='SOURCE_ONLY_FIXED_MINUS64_WARM_TARIFF',rows=rows,whole_bill=cert['flops'],gross_saved_flops=saved,gross_pct=100*saved/cert['flops'],sample8_proxy_flops=selection,sample8_proxy_pct=100*selection/cert['flops'],net_before_gathers_cold_sort_scatter=saved-selection,net_pct_before_gathers_cold_sort_scatter=100*(saved-selection)/cert['flops'],limits=['TRAIN16 only, no model calls. Matmul/projection actualFNP; unsupported virtual sum/gate weighting priced explicitly as10*c2 float32 operations.','Exactly incumbent c1.5 count minus64 at each existing changed read. No band/rank/sample sweep.','Selector charged8xKxC2 read plus two-dimensional sample projection and gate-square weight. Initial sample basis and gathers/sort/scatter/cold route debt omitted, so net is optimistic.','Actual refined Q cannot select its own prefix without computing that prefix first; selector must use a pre-read proxy.','Full native bill and accuracy remain unknown.'])
p=D/'FIXED_MINUS64_TARIFF_v2.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ('rows','limits')}));print(hashlib.sha256(p.read_bytes()).hexdigest())
