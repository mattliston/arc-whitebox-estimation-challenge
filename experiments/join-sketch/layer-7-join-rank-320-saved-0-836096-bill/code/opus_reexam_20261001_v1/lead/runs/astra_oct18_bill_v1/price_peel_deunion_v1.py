"""Current v61j full-wrapper warm public0.12.3 metadata; no numerical arrays/model."""
from pathlib import Path
import json,resource,time,hashlib,gc
D=Path(__file__).resolve().parent;R=D.parent;t0=time.monotonic()
source=R/'opus_fix336687_v1/candidate_v61j.py';pre=R/'astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py'
s=pre.read_text().split('def make_bini')[0].replace("p.SOURCE=O/'lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py'",'p.SOURCE=Path('+repr(str(source))+')')
s=s.replace("    if name in ('copy','astype')", "    if name in ('put','put_along_axis'):\n        p.CALLS[name]+=1;return None\n    if name=='arange':\n        p.CALLS[name]+=1;return p.virtual((len(range(*map(int,args))),),kwargs.get('dtype','int64'))\n    if name in ('copy','astype')")
ns={'__file__':str(pre),'__name__':'peel_current_metadata'};exec(s,ns)
M,fn,fs,arena,arr,measure=(ns[k]for k in('M','fn','fs','arena','arr','measure'))
# Existing frame geometry and complete compact-alias binders are preserved.
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 M._ASTRA_SM40_ENV._SCRATCH=M._SCRATCH
 M._astra_sm40_bind_universal(M._ASTRA_SM40_ENV,arena)
 M._smx_bind();M._peel_bind_read_io()
 M._YR7_ACTIVE[0]=True
 def bind(key):
  kind,L,r,c=key;e=M._ipt_side(kind,L,r,c);slot=e[2]
  if type(slot)is list and slot[1]is None:M._l3o_pairs(slot,L)if kind==2 else M._l3_quad(slot,kind,L)
 M._LB_SETUP[0]=True;M._LVF_EAGER[0]=True
 for key in M._PB_SINGLE[:128]:bind(key)
 M._LB_SETUP[0]=False;M._LVF_EAGER[0]=False;M._lb_make_boxes();M._gcv_prepare(280)
 M._LVF_EAGER[0]=True
 for key in M._PB_SINGLE[128:192]:bind(key)
 M._LVF_EAGER[0]=False
u=json.loads((R/'astra_crunch_compute_v1/small_stack_v25a_v1/UNION_PRICE_v2.json').read_text())
hist=json.loads((D/'PEEL_DEUNION_HISTORICAL_PRICE_v1.json').read_text())
rows=[];normal_take=M._peel_take;normal_xsg=M._xsg_rows;normal_inner=M._cp_slr_inner_sq
for row in hist['rows']:
 M._xsg_rows=normal_xsg;M._cp_slr_inner_sq=normal_inner
 kind=row['kind'];shape=row['shape'];m,k,n,extra=shape
 if not row['count']:continue
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  if kind=='push':
   q=extra;w=arr((1024,m));prev=fn.empty((k,),dtype=fn.int64);selected=fn.empty((q,),dtype=fn.int64);p0=arr((k,n+2*q));base=arr((1024,q));t=arr((q,));ss=arr((q,));newborn=(n,selected,base,t,ss)
   scope=M._w5_port_new(w)if m==1024 else None
   if scope is not None:M._w5_port_begin(scope,w,6,w,prev)
   M._W5_PORT_SCOPE=scope
   basecall=lambda:M._astra_bcc_newborn_impl(w,p0,prev,newborn)
   peelcall=lambda:M._JUN_PARENT_NEWBORN(w,p0,prev,newborn)
  else:
   rank=extra;old=int(rank>0);left=arr((m,k+old));right=arr((n,k));lam=arr((k+old,));indices=fn.empty((n,),dtype=fn.int64)
   inner=arr((m,rank));basis=arr((n,rank));state=dict(columns=old,pre_basis=arr((1024,rank)))
   M._cp_slr_inner_sq=lambda *args:inner;M._xsg_rows=lambda *args:basis
   basecall=lambda:M._PEEL_PARENT_READ(left,right,lam,indices,state)
   peelcall=lambda:M._peel_read(left,right,lam,indices,state)
 def resetcap():M._PEEL_CALL_BUDGET.update(left=750,spent=0,accepted=0,rejected=0)
 M._peel_take=normal_take;resetcap();measure(basecall);resetcap();baseprice=measure(basecall)
 # Metadata assumes target fixed route admitted; candidate must use its own static whitelist/cap.
 M._peel_take=lambda kind,key:True;measure(peelcall);peelprice=measure(peelcall);M._peel_take=normal_take
 rows.append(dict(kind=kind,shape=shape,base=baseprice,peel=peelprice,gain=baseprice['flops']-peelprice['flops'],request_delta=peelprice['requests']-baseprice['requests'],historical_gain=row['gain'],historical_request_delta=row['request_delta'],count=row['count']))
 print(json.dumps({k:rows[-1][k]for k in('kind','shape','gain','request_delta')}),flush=True)
 if kind=='push':
  if scope is not None:M._w5_port_end_layer(scope)
  M._W5_PORT_SCOPE=None
 del basecall,peelcall;M._BH_HOLD.clear();gc.collect()
report=dict(status='CURRENT_V61J_PUBLIC0123_WARM_METADATA',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sdk=fs.__version__,rows=rows,seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,limitations=['Historical eight roster coverage, not a current full census.','Warm costs only: initial family/view binding excluded, must prebind and measure in setup.','Read old-inner stub common to both; reconstruction cost excluded equally.','Request proxy counts wrapper calls/views, not residual instructions.','Fixed whitelist selection still required; no efficacy claim.'])
(D/'PEEL_DEUNION_CURRENT_PRICE_v1.json').write_text(json.dumps(report,indent=2)+'\n')
