"""Bound remaining historical fixed-list fusion; metadata, no model."""
from pathlib import Path
import hashlib,json,resource,time
D=Path(__file__).resolve().parent;R=D.parents[1];t0=time.monotonic()
source=R/'astra_crunch_runtime_v1/failure336678_v1/candidate_gen2_read_off_fusion192_v23c.py';pre=D.parent/'bini_apa_v1/price_outer_232_v3.py'
text=pre.read_text().split('def make_bini')[0].replace("p.SOURCE=O/'lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py'",'p.SOURCE=Path('+repr(str(source))+')')
ns={'__file__':str(pre),'__name__':'fusion_suffix_metadata'};exec(text,ns)
M,fn,fs,arena,measure=(ns[k]for k in('M','fn','fs','arena','measure'))
def bind(key):
 kind,L,r,c=key;e=M._ipt_side(kind,L,r,c);slot=e[2]
 if type(slot)is list and slot[1]is None:M._l3o_pairs(slot,L)if kind==2 else M._l3_quad(slot,kind,L)
 return e
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 M._LB_SETUP[0]=True;M._LVF_EAGER[0]=True
 for key in M._PB_SINGLE[:128]:bind(key)
 M._LB_SETUP[0]=False;M._LVF_EAGER[0]=False;M._lb_make_boxes();M._gcv_prepare(280)
 M._LVF_EAGER[0]=True
 for key in M._PB_SINGLE[128:192]:bind(key)
rows=[]
for i,key in enumerate(M._PB_SINGLE[192:],193):
 cost=measure(lambda:bind(key));rows.append(dict(index=i,key=key,cost=cost))
old=json.loads((R/'astra_crunch_runtime_v1/s2tr_peel_v1/FUSION_ACTUAL_CENSUS_v1.json').read_text())
front=[]
for prefix in(224,280,384,len(M._PB_SINGLE)):
 chosen=rows[:prefix-192];keys={tuple(x['key'])for x in chosen};oldrows=[]
 for x in old['rows']:
  hit=[p for p in x['priced_keys']if tuple(p['key'])in keys]
  oldrows.append(dict(rank=x['rank'],historical_calls_saved=sum(p['gross']for p in hit),historical_keys=len(hit)))
 front.append(dict(prefix=prefix,additional_setup_calls=sum(x['cost']['requests']for x in chosen),additional_setup_flops=sum(x['cost']['flops']for x in chosen),historical_v11_savings_not_current=oldrows))
oldremain=[];allkeys=set(M._PB_SINGLE)
for x in old['rows']:
 oldremain.append(dict(rank=x['rank'],all_unfused_historical=x['gross_max'],remaining_after_current192=sum(p['gross']for p in x['priced_keys']if tuple(p['key'])not in M._PB_SINGLE[:192]),outside_entire_fixedlist=sum(p['gross']for p in x['priced_keys']if tuple(p['key'])not in allkeys)))
out=dict(status='METADATA_CEILING_ONLY',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,frontier=front,historical_remaining=oldremain,models=0,seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Historical v11 per-key arithmetic-use census is not a current v23c whole-network projection; current read fallback changes paths.','Current-source setup costs are exact public metadata tariff, not measured startup time.','No new candidate or model; no score-dependent routing.'])
with(D/'FUSION_SUFFIX_PRICE_v1.json').open('x')as f:json.dump(out,f,indent=2)
print(json.dumps({k:out[k]for k in('frontier','historical_remaining','seconds','maxrss')}))
