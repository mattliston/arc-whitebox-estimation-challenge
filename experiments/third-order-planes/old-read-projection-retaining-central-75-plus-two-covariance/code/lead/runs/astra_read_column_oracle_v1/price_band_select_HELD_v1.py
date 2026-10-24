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

def selection_price(a,band):
 k=a['young_k']
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  A=fnp.empty((8,k),dtype=fnp.float32);B=fnp.empty((band,k),dtype=fnp.float32);Q=fnp.empty((8,2),dtype=fnp.float32);S=fnp.empty((2,k),dtype=fnp.float32)
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:
  Y=A@B.T;res=Y-Q@(S@B.T);squared=res*res
 return int(ctx.flops_used)+8*band+2*band,len(ctx._op_log)+3
for a in cert['capture_metadata']:
 l=a['layer'];c=a['c15']
 if c>=a['c2']:continue
 core=c-96;band=min(160,a['c2']-core)
 before=price(a,c);after=price(a,core)
 extra=dict(a);extra.update(old_rank=0,old_columns=0,c2=32)
 exact32=price(extra,32)['read_flops']
 sample,ops=selection_price(a,band)
 # Extra32 exact output substitutes already-paid projected young values. Debit
 # gathering/scattering all m*32 values and gathering32*K right values.
 transfer=a['m']*32*2+32*a['young_k']
 rows.append(dict(layer=l,original=c,core=core,band=band,incumbent_flops=before['flops'],core_and_tail_flops=after['flops'],extra32_read_flops=exact32,sample8_flops=sample,explicit_transfer=transfer,saving=before['flops']-after['flops']-exact32-sample-transfer))
 print(l,rows[-1]['saving'],exact32,sample,flush=True)
saved=sum(x['saving']for x in rows)
out=dict(status='SOURCE_ONLY_FIXED_CORE96_BAND160_SELECT32_TARIFF',rows=rows,whole_bill=cert['flops'],saved_flops=saved,saving_pct=100*saved/cert['flops'],limits=['Existing TRAIN16 shapes; no model calls.','Fixed8 sample rows; guaranteedcore=c15-96, choose32 amongnextmin160, residualQ fromcoreprefix once.','Mainprefix and extra32 youngread priced through actual inherited dispatch; extra32 hasNOoldinner. Full oldtail already included in core price.','Sample8 direct read minus Qsample(SBsample) included; explicit gathers/scatter included.','Sorting, sensitivity weights, basis/sample gathers and cold/route debt remain additional; a planning ceiling, not actualbill qualification.'])
p=D/'CORE96_BAND160_SELECT32_TARIFF_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('rows','limits')}));print(hashlib.sha256(p.read_bytes()).hexdigest())
