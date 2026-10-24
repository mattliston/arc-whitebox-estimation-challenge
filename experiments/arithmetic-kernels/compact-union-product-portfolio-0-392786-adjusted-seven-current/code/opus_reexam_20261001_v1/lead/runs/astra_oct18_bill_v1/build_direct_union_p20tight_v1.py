"""Freeze public tariff Pareto choices; retain every union column and old physics."""
import ast, hashlib, json
from pathlib import Path
D=Path(__file__).resolve().parent
parent=D/'candidate_v61j_bookkeeping_algebra_HELD_v2.py'
output=D/'candidate_v61j_bookkeeping_union_p20tight_HELD_v1.py'
prices=[D/n for n in ('PEEL_UNION_GEOMETRY_PRICE_v2.json','PEEL_UNION_FULLBODY_PRICE_v2.json','PEEL_UNION_CURRENT7_PRICE_v1.json','PEEL_UNION_CUSTOM_PRICE_v1.json','PEEL_UNION_P20_PRICE_v1.json')]
data=[json.loads(p.read_text())for p in prices]
price_rows=[x for table in data[:3] for x in table['rows']]
new_families={}
custom_tasks=json.loads((D/'PEEL_UNION_CUSTOM_TASKS_v1.json').read_text())['tasks']
for i,(row,task)in enumerate(zip(data[3]['rows'],custom_tasks)):
    p=json.loads((D/('PEEL_UNION_CUSTOM_PART_v3_%03d.json'%i)).read_text())
    if p['arena_extra_bytes']==0 and all(a==b for _,a,b in p['owner_growth']):
        price_rows.append(row);new_families[task['family'][0]]=tuple(task['family'])
p20_tasks=json.loads((D/'PEEL_UNION_P20_TASKS_v1.json').read_text())['tasks']
for row,task in zip(data[4]['rows'],p20_tasks):
    if task['family'][3]<=5 and task['family'][4]<=5 and task['family'][5]<=21:
        price_rows.append(row);new_families[task['family'][0]]=tuple(task['family'])
eligible=[x for x in price_rows if x['gain']>0 and
          (x['request_delta']<=0 or x['gain']>=500_000*x['request_delta'])]
groups={}
slow={(736,768,6400),(1024,992,8064),(1024,992,8128)}
for row in eligible:
    m,k,n,q=row['shape'];assert(m,k,n)not in slow
    key=(m,k,n+q);plan=list(row['plan']);plan[5]=tuple(tuple(x)for x in plan[5]);plan=tuple(plan)
    assert all(0<a<=b for a,b in zip(plan[2:5],key))
    option=(int(row['gain']),max(0,int(row['request_delta'])),plan)
    groups.setdefault(key,set()).add(option)
rules={}
for key,options in sorted(groups.items()):
    # Keep only tariff/call Pareto choices. At runtime choose highest gain that
    # fits the remaining fixed request allowance; no network values are read.
    frontier=[v for v in options if not any(w[0]>=v[0] and w[1]<=v[1] and
              (w[0]>v[0] or w[1]<v[1])for w in options)]
    rules[key]=tuple(sorted(frontier,key=lambda x:(-x[0],x[1],x[2])))
bind_plans=dict(enumerate(sorted({x[2]for options in rules.values()for x in options})))
used_families=tuple(sorted(new_families[name]for name in {p[1]for p in bind_plans.values()}if name in new_families))
arena_elements=max([391602176]+[7**f[1]*f[2]*(f[3]*f[4]+f[4]*f[5]+f[3]*f[5])for f in used_families])
s=parent.read_text();tree=ast.parse(s)
binder=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_rf192_bind')
old=ast.get_source_segment(s,binder)
new=old.replace('live = {plan[1] for plan in _SM12_PLANS.values()}',
    'live = {plan[1] for plans in (_SM12_PLANS, _O18_UNION_BIND_PLANS) for plan in plans.values()}')
new=new.replace('_PEEL_PUSH_PLANS, _PEEL_READ_PLANS = {}, {}',
    '_PEEL_PUSH_PLANS, _PEEL_READ_PLANS = _O18_UNION_BIND_PLANS, {}')
assert new!=old and new.count('_O18_UNION_BIND_PLANS')==2
s=s.replace(old,new,1)
node=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_astra_bcc_newborn_impl'][-1]
newborn=ast.get_source_segment(parent.read_text(),node)
debit="""    # Preserve the old scientific cap debit even though the union replaces this call.
    if _SMX_ON and (m, k, n) in _PEEL_PUSH_PLANS:
        _peel_take('push', (m, k, n))
"""
assert newborn.count(debit)==1 and newborn.count('    product = _mm(wp, rhs)')==1
newborn=newborn.replace(debit,'').replace('    product = _mm(wp, rhs)',
                                       '    product = _o18_union_product(wp, rhs)')
# Format checks recognize only the two certified owner lengths. Legacy packed
# offsets and reshapes remain byte-for-byte parent geometry.
guard_changes=[]
parent_text=parent.read_text();parent_lines=parent_text.splitlines()
functions=[n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
for lineno,line in enumerate(parent_lines,1):
    if 'int(_SCRATCH[0].shape[0])' in line and '391602176' in line:
        owners=[f for f in functions if f.lineno<=lineno<=f.end_lineno]
        owner=min(owners,key=lambda f:f.end_lineno-f.lineno)
        guard_changes.append(dict(line=lineno,function=owner.name,before=line.strip(),after='int(_SCRATCH[0].shape[0]) in (391602176, _O18_UNION_ARENA_ELEMENTS)'))
assert len(guard_changes)==8
for oldguard in ('int(_SCRATCH[0].shape[0]) == 391602176','int(_SCRATCH[0].shape[0])==391602176'):
    s=s.replace(oldguard,'int(_SCRATCH[0].shape[0]) in (391602176, _O18_UNION_ARENA_ELEMENTS)')
assert s.count('<= 391602176 and P in')==1
s=s.replace('<= 391602176 and P in','<= _PACKED_ELEMENTS and P in',1)
legacy_changes=[]
for name in ('_bh_o_frame','_ul_mode'):
    node0=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    oldlegacy=ast.get_source_segment(parent.read_text(),node0)
    assert oldlegacy.count('int(arena.shape[0])')==1
    newlegacy=oldlegacy.replace('int(arena.shape[0])','min(int(arena.shape[0]), 391602176)')
    assert oldlegacy in s;s=s.replace(oldlegacy,newlegacy,1)
    legacy_changes.append(dict(function=name,line=node0.lineno,change='Legacy tail availability capped at unchanged391602176-element prefix.'))
fragment='''

# The one existing scratch owner grows; all legacy packed prefixes stay fixed.
_O18_UNION_ARENA_ELEMENTS = __ARENA__
_PACKED_ELEMENTS = _O18_UNION_ARENA_ELEMENTS
_O18_UNION_NEW_FAMILIES = __FAMILIES__
for _o18_family in _O18_UNION_NEW_FAMILIES:
    _smx_register(_o18_family)

# Exact Oct18 direct-union peel: public-shape tariff choices only. No columns
# removed, no covariance/calibration change; parent fringes preserve content.
_O18_UNION_RULES = __RULES__
# The setup topology consumes only these values. Every alternative is prebound
# into existing shared owners; integer keys avoid dropping same-shape variants.
_O18_UNION_BIND_PLANS = __BINDS__
_O18_PEEL_STATE = dict(left=500, push=0, read=0, rejected=0, selected=[])
_O18_UNION_PARENT_PREDICT = Estimator.predict

def _o18_union_product(a, b):
    key = (int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))
    options = _O18_UNION_RULES.get(key, ()) if _SMX_ON else ()
    for gain, cost, plan in options:
        if cost <= _O18_PEEL_STATE['left']:
            _O18_PEEL_STATE['left'] -= cost
            _O18_PEEL_STATE['push'] += 1
            _O18_PEEL_STATE['selected'].append((key, cost, plan[2:5]))
            return _peel_product(a, b, plan)
    if options:
        _O18_PEEL_STATE['rejected'] += 1
    return _mm(a, b)

__NEWBORN__

def _o18_union_predict(self, mlp, budget):
    _O18_PEEL_STATE.update(left=500, push=0, read=0, rejected=0, selected=[])
    return _O18_UNION_PARENT_PREDICT(self, mlp, budget)

Estimator.predict = _o18_union_predict
'''.replace('__ARENA__',repr(arena_elements),1).replace('__FAMILIES__',repr(used_families),1).replace('__RULES__',repr(rules),1).replace('__BINDS__',repr(bind_plans),1).replace('__NEWBORN__',newborn,1)
s+=fragment;compile(s,str(output),'exec')
rollout=[]
for net in data[2]['networks']:
    left=500;gain=0;hits=[]
    for m,k,n,q in net['shapes']:
        for value,cost,plan in rules.get((m,k,n+q),()):
            if cost<=left:
                left-=cost;gain+=value;hits.append(dict(shape=[m,k,n,q],gain=value,cost=cost,plan=plan));break
    rollout.append(dict(rank=net['rank'],bill_gain_pct=100*gain/net['flops'],request_debit=500-left,hits=hits))
receipt=dict(status='HELD_UNTESTED_NATIVE',parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
             candidate=str(output),sha256=hashlib.sha256(s.encode()).hexdigest(),
             price_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in prices},
             rules=len(rules),alternatives=sum(map(len,rules.values())),bound_plans=len(bind_plans),
             rollout=rollout,mean_union_only_bill_gain_pct=sum(x['bill_gain_pct']for x in rollout)/len(rollout),
             cap=500,min_gain_per_positive_call=500000,accuracy_claim=False,arena_elements=arena_elements,arena_extra_bytes=4*(arena_elements-391602176),new_families=used_families,format_guard_changes=guard_changes,legacy_prefix_checks=legacy_changes,
             limitations=['Bookkeeping incremental bill not included in projection.','Warm metadata only; cold binds, real requests, accuracy and runtime untested.','Own prebind list required before release.'])
for name,body in [('fragment_direct_union_p20tight_v1.py',fragment),(output.name,s),('DIRECT_UNION_P20TIGHT_BUILD_v1.json',json.dumps(receipt,indent=2)+'\n')]:
    with(D/name).open('x')as f:f.write(body)
print(json.dumps(receipt))
