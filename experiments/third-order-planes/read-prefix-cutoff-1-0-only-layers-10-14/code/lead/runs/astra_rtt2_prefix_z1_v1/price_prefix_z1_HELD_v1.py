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
with np.load(R/'native_16_capture_capture.npz',allow_pickle=False)as z:
 for a in cert['capture_metadata']:
  l=a['layer'];c=a['c15']
  if f'L{l}_response_mu'not in z:continue
  alpha=z[f'L{l}_response_mu']/np.sqrt(z[f'L{l}_response_v']);drop=int(np.sum((alpha>=-1.5)&(alpha< -1.)))
  lo=((c-31-drop+31)//32)*32;hi=((c-drop+31)//32)*32
  prices={x:price(a,x)for x in sorted(set((c,lo,hi)))};base=prices[c]['flops']
  rows.append(dict(layer=l,original=c,dropped_band_count=drop,prefix_low=lo,prefix_high=hi,prices=prices,saving_at_low=base-prices[lo]['flops'],saving_at_high=base-prices[hi]['flops']))
  print(l,lo,hi,rows[-1]['saving_at_low'],rows[-1]['saving_at_high'],flush=True)
savings=[sum(r[key]for r in rows)for key in('saving_at_low','saving_at_high')]
out=dict(status='SOURCE_ONLY_WARM_PRICE_BOUNDS',rows=rows,whole_saved_flops_lowprefix=savings[0],whole_saved_flops_highprefix=savings[1],saving_pct_lowprefix=100*savings[0]/cert['flops'],saving_pct_highprefix=100*savings[1]/cert['flops'],source_sha256=p.SOURCE_SHA,limits=['TRAIN16 geometries only; no new model/capture.','c1 raw counts reconstructed from saved response alpha band and incumbentceil32 count; two exact rounding endpoints, no estimate substituted for true count.','L1 has no saved tail/response capture and is excluded: its possible new savings/error are not estimated.','Includes actual warm prefix dispatch, full shape-changing QR power products, exact old tail, rank2 young tail, concatenation. Same-sized QR, Q.T A and old-inner work cancel.','Native cold/cache/route changes and full bill remain actual gates.'],no_models=True,no_parameter_sweep=True)
q=D/'PREFIX_Z1_TRAIN16_TARIFF_BOUNDS_v1.json'
with q.open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k not in('rows','limits')}));print('receipt',hashlib.sha256(q.read_bytes()).hexdigest())

