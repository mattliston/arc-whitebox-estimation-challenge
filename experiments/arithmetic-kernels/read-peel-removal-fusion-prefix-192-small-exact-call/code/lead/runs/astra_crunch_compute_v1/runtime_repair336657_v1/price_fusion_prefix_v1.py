"""Current-source fixed historical fusion-prefix expansion; metadata only."""
from pathlib import Path
import ast,hashlib,json,resource,time
D=Path(__file__).resolve().parent;R=D.parents[1];t0=time.monotonic()
source=R/'astra_crunch_runtime_v1/failure336657_v1/candidate_setup_freeze_v22b.py'
pre=D.parent/'bini_apa_v1/price_outer_232_v3.py'
text=pre.read_text().split('def make_bini')[0].replace("p.SOURCE=O/'lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py'",'p.SOURCE=Path('+repr(str(source))+')')
ns={'__file__':str(pre),'__name__':'fusion_prefix_metadata'};exec(text,ns)
M,fn,fs,arena,measure=(ns[k]for k in('M','fn','fs','arena','measure'))
def bind(key):
 kind,L,rows,cols=key
 e=M._ipt_side(kind,L,rows,cols);slot=e[2]
 if type(slot)is list and slot[1]is None:
  M._l3o_pairs(slot,L)if kind==2 else M._l3_quad(slot,kind,L)
 return e
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 M._LB_SETUP[0]=True;M._LVF_EAGER[0]=True
 for key in M._PB_SINGLE[:128]:bind(key)
 M._LB_SETUP[0]=False;M._LVF_EAGER[0]=False
 M._lb_make_boxes();M._gcv_prepare(280)
rows=[]
for i,key in enumerate(M._PB_SINGLE[128:224],129):
 M._LVF_EAGER[0]=True
 result=measure(lambda:bind(key));M._LVF_EAGER[0]=False
 kind,L,r,c=key;e=(M._IPT_A,M._IPT_B,M._IPT_C)[kind][(L,r,c)]
 rows.append(dict(index=i,key=key,setup=result,fused=e[3]is not None,minimum_calls_saved_per_use=L if kind<2 else 2*(L-1),logical_tree_elements=7**L*(r//(1<<L))*(c//(1<<L))))
seq=[]
for line in (R/'astra_crunch_runtime_v1/failure336657_v1/natural_v22b_sequence_v1/rows.jsonl').read_text().splitlines():
 x=json.loads(line);new=set()
 for kind,name in enumerate(('_IPT_A','_IPT_B','_IPT_C')):
  new.update((kind,*ast.literal_eval(k))for k in x['cache_delta'][name]['added'])
 seq.append(dict(rank=x['rank'],new_keys=new,requests=x['requests']))
front=[]
for prefix in (144,160,176,192,208,224):
 chosen=rows[:prefix-128];keys={tuple(x['key']):x for x in chosen}
 lower=[]
 for x in seq:
  used=x['new_keys']&keys.keys();credit=sum(keys[k]['minimum_calls_saved_per_use']for k in used)
  lower.append(dict(rank=x['rank'],new_key_count=len(used),recurring_call_lower_bound=credit,preparation_calls_removed=sum(keys[k]['setup']['requests'] for k in used)))
 front.append(dict(prefix=prefix,setup_calls=sum(x['setup']['requests']for x in chosen),setup_flops=sum(x['setup']['flops']for x in chosen),current_natural10_lower_bounds=lower))
out=dict(status='METADATA_ONLY',models=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,frontier=front,seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Current natural10 records newly created keys only; one use per key is a conservative arithmetic-call lower bound, not whole-network projected savings.','Fixed existing historical prefix; no score- or fixture-dependent routing.','The current packed/boxed key class remains unchanged; _LB_SETUP stays false for additional views.','Setup count excludes retained-handle facade Python and free batching; hardware qualification required.','Current source already disables large-tree fusion limit; additional fused large B/C trees can increase kernel work.'])
with(D/'FUSION_PREFIX_PRICE_v1.json').open('x')as f:json.dump(out,f,indent=2)
print(json.dumps({k:out[k]for k in('frontier','seconds','maxrss')}))
