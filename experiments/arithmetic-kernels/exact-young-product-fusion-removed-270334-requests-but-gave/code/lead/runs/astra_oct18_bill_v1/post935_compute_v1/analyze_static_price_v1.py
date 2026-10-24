import ast,json,statistics,hashlib
from pathlib import Path
R=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs')
D=R/'astra_oct18_bill_v1';O=D/'post935_compute_v1';O.mkdir(exist_ok=True)
helper=(D/'analyze_compact_portfolio_v3.py').read_text().split('geometry=[];allfront=[]')[0].replace('OUT.mkdir(exist_ok=False)','pass')
scope={'__file__':str(D/'analyze_compact_portfolio_v3.py')};exec(compile(helper,'static-assembly','exec'),scope)
src=R/'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
lit={}
for n in ast.parse(src.read_text()).body:
 if isinstance(n,ast.Assign):
  for t in n.targets:
   if isinstance(t,ast.Name):
    try:lit[t.id]=ast.literal_eval(n.value)
    except (ValueError,TypeError):pass
current=lit['_O18_UNION_RULES'];grid=scope['grid'];remap=scope['remap'];grids={grid(p)for options in current.values()for g,c,p in options};fams={p[1]for options in current.values()for g,c,p in options}
assert len(grids)==10
raw={key:tuple((g,c,remap(p))for g,c,p in vs)for key,vs in scope['all_rules'].items()}
samegrid={key:tuple(v for v in vs if grid(v[2])in grids)for key,vs in raw.items()}
samefamily={key:tuple(v for v in vs if v[2][1]in fams)for key,vs in raw.items()}
rows=list(map(json.loads,(R/'astra_oct18_lead_v1/compact10_repair_qualification_v1/science32/candidate32/rows.jsonl').read_text().splitlines()))
keys=lambda r:[(m,k,n+q)for m,k,n,q in r['jun_stats']['shapes'][-15:]]
roll=scope['rollout'];base=[roll(current,keys(r),450)for r in rows]
for r,b in zip(rows,base):assert [(list(k),c,list(p[2:5]))for k,g,c,p in b[2]]==[(k,c,p)for k,c,p in r['oct18_peel_state']['selected']]
def oracle(rules,ks,cap):
 states={0:0}
 for k in ks:
  nxt=dict(states)
  for c,g in states.items():
   for dg,dc,p in rules.get(k,()):
    if c+dc<=cap:nxt[c+dc]=max(nxt.get(c+dc,-1),g+dg)
  states=nxt
 g,neg=max((g,-c)for c,g in states.items());return g,-neg
report={}
for label,rules in [('same10grids',samegrid),('samefamilies_moregrids',samefamily)]:
 for cap in (450,500):
  vals=[oracle(rules,keys(r),cap)for r in rows];inc=[100*(v[0]-b[0])/r['flops']for r,v,b in zip(rows,vals,base)]
  report[f'{label}_oracle{cap}']=dict(incremental_bill_pct=statistics.mean(inc),conditional_score=1.4468565569762618e-9*(1-statistics.mean(inc)/100),mean_positive_debit=statistics.mean(v[1]for v in vals),rank35_debit=vals[0][1],rank200_debit=vals[2][1],grid_count=len({grid(p)for vs in rules.values()for g,c,p in vs}),note='Hindsight ceiling, not implementable allocation. No accuracy claim; current graded score anchor.')
# Fitted source model only, with all A/B/C transformations included; no SDK.
def bill(L,m,k,n):
 S=((7/4)**L-1)/.75
 return 2*m*k*n*(7/8)**L-m*n*(7/4)**L+.7669*m*k*S+.7668*k*n*S+1.5213*m*n*S+2.6307*(m*k+k*n+m*n)
fusion=[]
for old in (5,6):
 for new in (6,7):
  per=[]
  for r,b in zip(rows,base):
   U=[v[4:7]for v in r['product_geometries']if v[0]=='shared_old384'and v[2]=='_w5_port_call'and v[6]==384];J=r['jun_stats']['shapes'][-15:][6:];assert len(U)==len(J)==9
   chosen={tuple(k)for k,g,c,p in b[2]};gain=0;count=0;tailflops=0;equal=0;eligible=[]
   for L,((m,k,rank),(my,kj,n,q))in enumerate(zip(U,J),7):
    assert k==kj and m>=my
    if (my,k,n+q)in chosen:continue
    tail=(m-my)*384*(2*k-1);g=bill(old,m,k,384)-(bill(new,my,k,n+q+384)-bill(new,my,k,n+q))-tail
    gain+=g;tailflops+=tail;count+=1;equal+=m==my;eligible.append([L,m,my,k,n+q,g])
   per.append(dict(rank=r['rank'],eligible=count,equalrows=equal,tailMF=tailflops/1e6,model_gain_pct=100*gain/r['flops'],shapes=eligible))
  fusion.append(dict(old_depth=old,fused_depth=new,mean_model_gain_pct=statistics.mean(p['model_gain_pct']for p in per),mean_kernels_removed=statistics.mean(p['eligible']for p in per),per_network=per))
receipt=dict(status='STATIC_ONLY_NO_SDK_NO_GAIN_BANKED',source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),current_score=1.4468565569762618e-9,union_frontier=report,fusion_unpeeled_models=fusion,limits=['Fusion source fit ignores actual hybrid D7/HZ7 dispatch, padding and extra copy/concat/scatter overhead, so not certified tariff.','Fused width n+384 may cause additional packet and requests; removed kernel count does not imply request saving.','Only currently unpeeled union products are eligible in model, preserving existing union credits.','Two source-model depths are bounds/scenarios, not endpoint-selected policies.'])
p=O/'STATIC_PRICE_v1.json';p.open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'union':report,'fusion':[{k:v for k,v in x.items()if k!='per_network'}for x in fusion]},indent=2))
