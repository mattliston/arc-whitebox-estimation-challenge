"""Static geometry/tariff family restriction. No SDK or numerical imports."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import statistics

D = Path(__file__).resolve().parent
R = D.parent
ROOT = R/'astra_oct18_lead_v1'
OUT = D/'compact_portfolio_v3'
OUT.mkdir(exist_ok=False)

# Reuse the already audited price-table assembly, stopping before its rollouts
# and immutable output writes. This executes only stdlib JSON/AST processing.
assembly = (D/'analyze_union_request_allocation_v1.py').read_text().split('baseline = ')[0]
scope = {'__file__': str(D/'analyze_union_request_allocation_v1.py')}
exec(compile(assembly, '<static-price-assembly>', 'exec'), scope)
all_rules = scope['all_rules']
freeze = scope['freeze']

source = D/'request_backup_cap300_v1/candidate_request_backup_cap300_HELD_v1.py'
tree = ast.parse(source.read_text())
literal = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    literal[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
families = {f[0]: f for name in ('_O18_UNION_NEW_FAMILIES', '_O18_LAYOUT_FAMILIES', '_O18_MARGIN_FAMILIES') for f in literal[name]}
exact = {(f[1], f[2], *f[3:]): f[0] for name in ('_O18_LAYOUT_FAMILIES', '_O18_MARGIN_FAMILIES') for f in literal[name]}

def remap(plan):
    family = families[plan[1]]
    _, L, P, *_ = family
    axes = (3, 3, 6) if plan[0] == 'signed_executable_0' else (3, 6, 3)
    leaf = tuple(-(-v//(a*(1<<L))) for v, a in zip(plan[2:5], axes))
    target = exact.get((L, P, *leaf), plan[1])
    return (plan[0], target) + plan[2:]

def clean(rules):
    return {key: tuple(sorted({(g, c, remap(p)) for g,c,p in options if g >= 2000000*c},
                             key=lambda v: (-(v[0]-2000000*v[1]),v[1],-v[0],v[2])))
            for key, options in rules.items()}
current = clean(literal['_O18_UNION_RULES'])
# Same registered geometries; broaden only to previously priced alternatives.
priced = clean({key: all_rules.get(key, ()) for key in current})
run = ROOT/'cap300_science32_v1'
rows = [json.loads(line) for line in (run/'candidate32/rows.jsonl').read_text().splitlines()]
controls = {row['rank']: row for row in map(json.loads, (run/'control32/rows.jsonl').read_text().splitlines())}
data = [(row, [(m,k,n+q) for m,k,n,q in row['jun_stats']['shapes'][-15:]]) for row in rows]
assert len(data) == 32

def rollout(rules, keys, cap, allowed=None):
    left = cap
    gain = 0
    selected = []
    for key in keys:
        for g,c,p in rules.get(key, ()):
            if c <= left and (allowed is None or p[1] in allowed):
                gain += g
                left -= c
                selected.append((key,g,c,p))
                break
    return gain, cap-left, selected

baseline = [rollout(current,keys,300) for row,keys in data]
for (row,keys),(gain,cost,chosen) in zip(data,baseline):
    assert cost == 300-row['oct18_peel_state']['left'], (row['rank'],cost,row['oct18_peel_state'])
    assert len(chosen) == row['oct18_peel_state']['push']
    observed = [(tuple(key), cost, tuple(body)) for key,cost,body in row['oct18_peel_state']['selected']]
    replayed = [(key,c,tuple(plan[2:5])) for key,g,c,plan in chosen]
    assert observed == replayed, (row['rank'],observed,replayed)
best = 1.4515052706908603e-9
actual = json.loads((run/'RESULT32_v2.json').read_text())['all32']
anchor = actual['projected_adjusted_score']
basecost = {row['rank']: v[1] for (row,keys),v in zip(data,baseline)}

def summarize(rules, allowed, cap):
    values = [rollout(rules,keys,cap,allowed) for row,keys in data]
    # Anchor to actual cap300 science (including measured cold IO charges).
    # Only change the public tariff delta; keep that science's numerical error.
    increments = [(v[0]-b[0])/row['flops'] for (row,keys),b,v in zip(data,baseline,values)]
    projected = anchor*(1-statistics.mean(increments))
    costs = {row['rank']:v[1] for (row,keys),v in zip(data,values)}
    return dict(families=sorted(allowed),cap=cap,conditional_score=projected,
                adjusted_pct_vs336867=100*(projected/best-1),
                incremental_bill_gain_vs_cap300_pct=100*statistics.mean(increments),
                mean_call_debit=statistics.mean(v[1] for v in values),max_call_debit=max(v[1] for v in values),
                rank35_debit=costs[35],rank200_debit=costs[200],
                rank35_extra_calls=costs[35]-basecost[35],rank200_extra_calls=costs[200]-basecost[200],
                selected_plan_count=len({x[3] for v in values for x in v[2]}))

universe = sorted({p[1] for options in priced.values() for g,c,p in options})

# Price only the prespecified fixed tariff grid; no endpoint optimization.
raw_priced={key:tuple((g,c,remap(p))for g,c,p in all_rules.get(key,()))for key in current}
def with_penalty(value):
    return {key:tuple(sorted((v for v in options if v[0]>=value*v[1]),
             key=lambda v:(-(v[0]-value*v[1]),v[1],-v[0],v[2])))for key,options in raw_priced.items()}
def grid(plan):
    _,L,P,*_=families[plan[1]];u=1<<L
    axes=(3,3,6)if plan[0]=='signed_executable_0'else(3,6,3)
    return(plan[1],L,*tuple(-(-v//(a*u))*u for v,a in zip(plan[2:5],axes)),P)
def restrict(rules,allowedgrids):
    return{k:tuple(v for v in vs if grid(v[2])in allowedgrids)for k,vs in rules.items()}
def inventory_result(summary,rules,cap):
    keep=set()
    for options in rules.values():
        for left in range(cap+1):
            for g,c,p in options:
                if c<=left:keep.add(p);break
    return dict(summary,reachable_plan_count=len(keep),distinct_outer_grids=len({grid(p)for p in keep}),
                reachable_plans=sorted(keep))
def compact(x):return{k:v for k,v in x.items()if k not in['reachable_plans','allowed_grids']}
allfamilies=frozenset(universe)

# Protected oldSM12 plus five shared additions, including exact-layout restores.
protected={sh:(rec,fam,*sh)for sh,(rec,fam)in literal['_SM12_RULE'].items()}
for sh,p in literal['_O18_SHARED_PLANS'].items():
    fam=p[1]if sh in literal['_O18_SHARED_RESTORE_SHAPES']else literal['_O18_SHARED_FAMILY_MAP'].get(p[1],p[1])
    protected[sh]=(p[0],fam,*p[2:])
for name in('_SM12_FAMILIES','_O18_SHARED_FAMILIES'):
    families.update((f[0],f)for f in literal[name])
protected_families={p[1]for p in protected.values()}
def topology(v):
    plans=v['reachable_plans']; unionfamilies={p[1]for p in plans}
    allplans=list(plans)+list(protected.values());side=set();groups=set();allgrids=set()
    for p in allplans:
        _,L,P,*_=families[p[1]];u=1<<L;axes=(3,3,6)if p[0]=='signed_executable_0'else(3,6,3)
        cm,ck,cn=tuple(-(-x//(a*u))*u for x,a in zip(p[2:5],axes))
        allgrids.add((p[1],L,cm,ck,cn,P))
        for kind,(r,c)in enumerate(((cm,ck),(ck,cn),(cm,cn))):
            side.add((p[1],kind,L,P,r,c));groups.add((p[0],kind,r,c))
    return dict(v,live_family_frames=len(unionfamilies|protected_families),protected_families=sorted(protected_families),
                all_outer_grids=len(allgrids),all_side_geometries=len(side),all_group_geometries=len(groups))
geometry=[];allfront=[]
for penalty in(1000000,1500000,2000000,2500000,3000000):
    rules=with_penalty(penalty);gs=frozenset(grid(p)for vs in rules.values()for g,c,p in vs)
    for cap in(300,350,400,450,500):
        v=summarize(rules,allfamilies,cap);v['penalty_per_call']=penalty;allfront.append(v)
        allowed=frozenset()
        for step in range(min(18,len(gs))):
            proposals=[]
            for new in gs-allowed:
                trial=allowed|{new};rr=restrict(rules,trial);v=summarize(rr,allfamilies,cap)
                if v['rank35_extra_calls']<=50 and v['rank200_extra_calls']<=50:
                    proposals.append((v['conditional_score'],new,v,rr))
            if not proposals:break
            score,new,v,rr=min(proposals,key=lambda z:(z[0],z[1]));allowed=allowed|{new}
            v.update(penalty_per_call=penalty,allowed_grids=sorted(allowed),families=sorted({g[0]for g in allowed}))
            geometry.append(topology(inventory_result(v,rr,cap)))
geometry.sort(key=lambda v:v['conditional_score'])
half=sorted((v for v in geometry if v['reachable_plan_count']<=36),key=lambda v:v['conditional_score'])
v2=json.loads((D/'compact_portfolio_v2/RESULT_v2.json').read_text())
strict=[topology(v)for v in v2['best_geometry_portfolios']+v2['best_half_geometry']]
report=dict(status='STATIC_TARIFF_ONLY_REQUIRES_NEW7_AND32',extra35_200_call_allowance=50,protected_families=sorted(protected_families),
            best_geometry=geometry[:25],best_half_geometry=half[:25],strict_frontier_with_full_topology=strict,
            full_policies=allfront,baseline_score=anchor,
            notes=['SharedSM12families are counted even when their unionroutes are removed.',
                   '35/200extra50call allowance is only static screening; unchanged measured runtime limits remain mandatory.',
                   'Projection keeps cap300 actualnumericalerrorandIOtax; newnumeric/coldIOmustbe measured.'])
with(OUT/'RESULT_v3.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'best':[compact(v)for v in geometry[:2]],'half':[compact(v)for v in half[:2]],'protected':sorted(protected_families)},indent=2))
