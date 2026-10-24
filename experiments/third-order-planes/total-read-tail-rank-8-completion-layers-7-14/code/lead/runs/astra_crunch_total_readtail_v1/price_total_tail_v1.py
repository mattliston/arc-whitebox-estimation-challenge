"""Saved geometry, current weighted dispatch; optimistic rank8 total-tail ceiling."""
from pathlib import Path
import hashlib, importlib.util, json, time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
s=importlib.util.spec_from_file_location('virtual_tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py'
p.SOURCE_SHA='46e9315d1a3e563cf669f27ec499be487ca7a437ee942b7be44ea1fad682ad9f'
M,_=p.load();f=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 arena=f.empty((M._PACKED_ELEMENTS,),dtype=f.float32)
 M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None]
 M._bind_push6(arena)
 M._ASTRA_SM40_ENV._SCRATCH=M._SCRATCH
 M._astra_sm40_bind_universal(M._ASTRA_SM40_ENV,arena)
M._YR7_ACTIVE[0]=True;M._WINOGRAD=True;M._AB_ON=True
M._YR7_STATS=dict(calls=0,chunks=0,shapes=[],routed_shapes=[],old_fused=0,max_arena_elements=0)
M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
CAP=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
CERT=json.loads((O/'lead/runs/astra_read_column_tail_rank2_v1/run_capture_train16_v1/native_16_capture.json').read_text())
META={a['layer']:a for a in CERT['capture_metadata']}
def meter(fn):
 with fs.BudgetContext(flop_budget=2**60,quiet=True) as ctx:out=fn()
 return out,dict(flops=int(ctx.flops_used),charged_ops=len(ctx._op_log),operations=ctx.summary_dict()['operations'])
def price(a,c,new):
 m,k,old,c2=[a[t] for t in ('m','young_k','old_columns','c2')]
 r=384 # Current S2tr uses rank384 at every old refresh; saved L10-12 had320.
 q=8 if new else 2;t=c2-c
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  left=f.empty((m,old+k),dtype=f.float32);right=f.empty((c,k),dtype=f.float32)
  lam=f.empty((old+k,),dtype=f.float32);idx=f.asarray(np.arange(c,dtype='int64'))
  inner=f.empty((m,r),dtype=f.float32);oldright=f.empty((c,r),dtype=f.float32)
  state=dict(columns=old,pre_basis=f.empty((1024,r),dtype=f.float32))
  Q=f.empty((m,q),dtype=f.float32);bt=f.empty((t,k),dtype=f.float32)
  Btail=f.empty((t,r),dtype=f.float32)
 M._cp_slr_inner_sq=lambda *args:inner
 M._xsg_rows=lambda *args:oldright
 M._TR_LAST[0]=None
 meter(lambda:M._RTT_PARENT_READ(left,right,lam,idx,state))
 prefix,read=meter(lambda:M._RTT_PARENT_READ(left,right,lam,idx,state))
 M._BH_HOLD.clear();M._TR_LAST[0]=None
 def complete():
  if not new:
   V=prefix.T@Q-oldright@(inner.T@Q)
   U=prefix@V-inner@(oldright.T@V)
  # For new, ALL basis preparation/refinement is deliberately free.
  # Parent's two QR calls also omitted: quantified as a small separate bound.
  scores=Q.T@left[:,old:]
  weighted=bt*lam[old:]
  projected=scores@weighted.T
  if new:projected=projected+(Q.T@inner)@Btail.T
  tail=Q@projected
  if not new:tail=tail+inner@Btail.T
  return f.concatenate((prefix,tail),axis=1)
 _,tail=meter(complete)
 return dict(total_flops=read['flops']+tail['flops'],charged_ops=read['charged_ops']+tail['charged_ops'],read=read,tail=tail)
rows=[];start=time.monotonic()
for layer in range(7,15):
 path=CAP/f'L{layer:02d}.npz'
 with np.load(path,allow_pickle=False) as z:
  c15=int(z['c15']);c2=int(z['c2']);c1=min(c2,((int(np.sum(z['gate']>=.15865525393145707))+31)//32)*32)
 a=META[layer];assert a['c15']==c15 and a['c2']==c2
 before=price(a,c15,False);after=price(a,c1,True)
 # A very conservative parent two-QR upper credit in float64, no candidate QR.
 # QR tariff itself is tiny; 100*m*2^2*8 exceeds it and QR/vector preparation.
 qr_credit=100*a['m']*4*8
 row=dict(layer=layer,m=a['m'],k=a['young_k'],c1=c1,c15=c15,c2=c2,current_old_rank=384,saved_old_rank=a['old_rank'],before=before,after_free_basis=after,saved_flops=before['total_flops']-after['total_flops'],parent_qr_generous_credit=qr_credit,capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('before','after_free_basis','capture_sha256')}),flush=True)
saved=sum(x['saved_flops'] for x in rows);credit=sum(x['parent_qr_generous_credit'] for x in rows)
doc=dict(status='CURRENT_WEIGHTED_TARIFF_SAVED_GEOMETRY_OPTIMISTIC_BOUND',rank=8,cutoff=1.0,layers=[7,14],source_sha256=p.SOURCE_SHA,rows=rows,optimistic_saved_flops=saved+credit,optimistic_saved_pct=100*(saved+credit)/244596689874,raw_saved_flops=saved,whole_current_STORE35_bill=244596689874,elapsed_s=time.monotonic()-start,models=0,limits=['Only historical TRAIN16 geometry; actual current native geometry may differ.','Current S2tr dispatch with public weighted tariffs; metadata numerical backend.','Old inner is exactly retained and common. Full exact prefix old outer retained. New tail projects total old+young read.','Rank8 basis is free; parent two QR calls generously credited; no candidate QR, power iteration, selection, extra binding/request debt.','Warm kernel costs after individual warmup. Not an observed native bill or request count.','Gate1.0 is fixed only at layers7-14 where old bank exists. No rank/cutoff sweep.'])
(D/'TARIFF_BOUND_v1.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps({k:v for k,v in doc.items() if k not in ('rows','limits')}),flush=True)
