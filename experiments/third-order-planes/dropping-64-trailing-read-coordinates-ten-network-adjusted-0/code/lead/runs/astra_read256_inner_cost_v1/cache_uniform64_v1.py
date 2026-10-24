import os,json,hashlib,resource
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS,0)==10
D=Path(__file__).resolve().parent;V=D.parent/'astra_smirnov_current24_virtual_v1';S=D/'candidate_uniform64_read_HELD_v1.py';source=(V/'price_v1.py').read_text().split('circuits=json.loads')[0];source=source.replace("p.SOURCE=O/'lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'","p.SOURCE=Path("+repr(str(S))+")");source=source.replace("p.SOURCE_SHA='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'","p.SOURCE_SHA="+repr(hashlib.sha256(S.read_bytes()).hexdigest()));g={'__file__':str(V/'price_v1.py')};exec(source,g)
M=g['M'];fnp=g['fnp'];fs=g['fs'];measure=g['measure'];rows=[]
def uv_make(rank,P):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):return(fnp.empty((256,P),dtype=fnp.float32),fnp.empty((rank,P),dtype=fnp.float32),((0,P),),256)
def operands(m,P):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):return fnp.empty((m,2*P),dtype=fnp.float32),fnp.empty((2*P,),dtype=fnp.float32)
P=5920;uv=uv_make(320,P)
with fs.BudgetContext(flop_budget=2**60,quiet=True):owner=fnp.empty((320,2*P),dtype=fnp.float32)
run=dict(owner=owner,uv=uv,generation=('fixed_base',P),cache={},bytes=0,stats=dict(generations=1,eligible=0,hits=0,misses=0,stores=0,copies=0,copy_elements=0,copied_bytes=0,max_resident_bytes=0,uncached=0,peak_cache_bytes=0,hit_slots={'u:0':0,'v:0':0,'v:4096':0,'ut:0':0,'vt:0':0,'u:4096':0}));M._UBC_RUN[0]=run;M._UBX_STATE.update(run=None,cache={},bytes=0);M._UBX_STATS.update(products=0,hits=0,stores=0,uncached=0,skipped=0,strip_hits=0)
for label,layer,rank,pp,m,on,use_base in [('base_cold',10,320,P,640,True,True),('base_hit',11,320,P,608,True,True),('base_lazy_hit',13,320,P,576,True,True),('recent_fallback',13,384,2400,576,True,False),('early_fallback',7,384,3072,800,True,False),('off_distinct_width',10,320,P,640,False,True),('outside_band',15,320,P,640,True,True)]:
 active=uv if use_base else uv_make(rank,pp);a,lam=operands(m,pp);M._UVP_CUR[0]=layer;M._ASTRA_READ64_ON=on;M._ASTRA_READ64_STATS['events'].clear();before=dict(M._UBX_STATS);answers=[];price=measure(lambda:answers.append(M._uvp_inner(a,lam,active)));after=dict(M._UBX_STATS);delta={k:after[k]-before[k]for k in before};assert answers[0].shape==(m,rank);assert M._UBX_CTX[0]is None and M._UBC_CONTEXT[0]is None;assert M._UBC_RUN[0]is run and run['owner']is owner and run['uv']is uv;assert M._ASTRA_READ64_STATS['events']==([(layer,m,pp,rank,rank-64,256)]if on and 7<=layer<=14 else [])
 if label in ('base_hit','base_lazy_hit'):assert delta['hits']>0 and delta['stores']==0
 if not use_base:assert delta['skipped']==2
 rows.append(dict(label=label,layer=layer,rank=rank,P=pp,m=m,on=on,price=price,stats_delta=delta,events=list(M._ASTRA_READ64_STATS['events'])))
# Exact cache tags distinguish vt256, ut256, and uncutvt320; no false sharing.
keys=list(M._UBX_STATE['cache']);key_text=repr(keys);assert "'vt'"in key_text and "'ut'"in key_text and '256'in key_text and '320'in key_text
M._UVP_CUR[0]=10;M._ASTRA_READ64_ON=True;M._ASTRA_READ64_STATS['events'].clear();a,lam=operands(640,P);original=M._shared_old384_mm
M._shared_old384_mm=lambda *args:(_ for _ in()).throw(RuntimeError('expected'))
try:
 try:measure(lambda:M._uvp_inner(a,lam,uv))
 except RuntimeError:pass
 else:raise AssertionError('exception not propagated')
 assert M._UBX_CTX[0]is None and M._ASTRA_READ64_STATS['events']==[]
finally:M._shared_old384_mm=original
M._UBC_RUN[0]=None;M._UBX_STATE.update(run=None,cache={},bytes=0);M._BH_HOLD.clear()
out=dict(passed=True,candidate_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),rows=rows,distinct_cache_key_repr=key_text,exception_context_cleared=True,retired=True,scope='Actual held source import and complete _uvp_inner with actual FNP virtualcachedbackend. Cachedbase hits acrossrow/layer changes, early/recentfallback, off/band guards, fullrank outputs, owneridentity andexceptioncontext. No model.',max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'CACHE_UNIFORM64_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(rows),max_rss_bytes=out['max_rss_bytes'],candidate_sha256=out['candidate_sha256'])))
