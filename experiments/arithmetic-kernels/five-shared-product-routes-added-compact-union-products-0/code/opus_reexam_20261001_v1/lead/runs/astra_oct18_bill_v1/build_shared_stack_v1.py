"""Add five current public-shape full products, retaining the tested union stack."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
parent=D/'candidate_v61j_bookkeeping_union_p20tight_HELD_v1.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='42af3fd042d029bcae9fdcf75e3eaad74c66dde09fb67df0fe1a6064bb7fed3f'
price=json.loads((D/'SHARED_PRODUCTS_PRICE_v2.json').read_text())
assert price['status']=='COMPLETE_WARM_METADATA'
rules={};families={};stats=[]
for row,part in zip(price['rows'],price['parts']):
 receipt=json.loads(Path(part).read_text())
 assert all(a==b for _,a,b in receipt['owner_growth'])
 gain=int(row['gain']);cost=max(0,int(row['request_delta']))
 if gain<=0 or cost>64 or (cost and gain<500000*cost):continue
 assert not row['contextual']
 shape=tuple(row['shape']);plan=list(row['plan']);plan[5]=tuple(tuple(x)for x in plan[5]);plan=tuple(plan)
 assert tuple(plan[2:5])==shape
 family=tuple(row['family']);assert family[1:3]==(5,40)
 assert 7**family[1]*family[2]*(family[3]*family[4]+family[4]*family[5]+family[3]*family[5])<=552950300
 rules[shape]=(gain,cost,plan);families[family[0]]=family;stats.append(row)
assert len(rules)==5
plans={shape:plan[:5]for shape,(_,_,plan)in rules.items()}
fragment='''

# Public-shape exact L5/P40 extension. Existing SM12 runtime rules are immutable;
# only its setup plan inventory is extended to prebind these five full products.
_O18_SHARED_RULES = __RULES__
_O18_SHARED_FAMILIES = __FAMILIES__
_O18_SHARED_PLANS = __PLANS__
assert not set(_O18_SHARED_RULES).intersection(_SM12_RULE)
for _o18_shared_family in _O18_SHARED_FAMILIES:
    _smx_register(_o18_shared_family)
_SM12_PLANS = dict(_SM12_PLANS, **{})
_SM12_PLANS.update(_O18_SHARED_PLANS)
_O18_SHARED_PARENT = _shared_old384_mm
_O18_SHARED_PARENT_PREDICT = Estimator.predict
_O18_SHARED_STATE = dict(left=64, accepted=0, rejected=0, selected=[])

def _shared_old384_mm(left, right):
    # Preserve every captured-reuse context and emergency route. The unchanged
    # parent owns existing SM12 keys, unsupported inputs and exhausted budgets.
    scope = _W5_PORT_SCOPE
    if (not _SMX_ON or not _AB_ON or not _WINOGRAD or _SCRATCH is None
            or _FAST_RESCUE_ON or _UBC_CONTEXT[0] is not None
            or _UBX_CTX[0] is not None or _ASTRA_CLUSTER_CTX[0] is not None
            or (scope is not None and scope.get('grant') is not None)
            or len(left.shape) != 2 or len(right.shape) != 2
            or left.dtype != 'float32' or right.dtype != 'float32'
            or int(left.shape[1]) != int(right.shape[0])):
        return _O18_SHARED_PARENT(left, right)
    key = (int(left.shape[0]), int(left.shape[1]), int(right.shape[1]))
    if key in _SM12_RULE:
        return _O18_SHARED_PARENT(left, right)
    option = _O18_SHARED_RULES.get(key)
    if option is not None:
        gain, cost, plan = option
        if cost <= _O18_SHARED_STATE['left']:
            _O18_SHARED_STATE['left'] -= cost
            _O18_SHARED_STATE['accepted'] += 1
            _O18_SHARED_STATE['selected'].append((key, cost))
            return _peel_product(left, right, plan)
        _O18_SHARED_STATE['rejected'] += 1
    return _O18_SHARED_PARENT(left, right)

def _o18_shared_predict(self, mlp, budget):
    _O18_SHARED_STATE.update(left=64, accepted=0, rejected=0, selected=[])
    return _O18_SHARED_PARENT_PREDICT(self, mlp, budget)

Estimator.predict = _o18_shared_predict
'''.replace('__RULES__',repr(rules),1).replace('__FAMILIES__',repr(tuple(sorted(families.values()))),1).replace('__PLANS__',repr(plans),1)
source=parent.read_text()+fragment
out=D/'candidate_v61j_union_shared_HELD_v1.py'
compile(source,str(out),'exec')
gain=sum(r['gain']*r['count']for r in stats)
receipt=dict(status='HELD_UNTESTED_NATIVE',parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
             candidate=str(out),sha256=hashlib.sha256(source.encode()).hexdigest(),
             rules=rules.__len__(),families=sorted(families.values()),plans=stats,
             cap=64,request_debit_rank35=sum(max(0,r['request_delta'])*r['count']for r in stats),
             rank35_only_credit_flops=gain,rank35_only_bill_gain_pct=100*gain/price['baseline_flops'],
             arena_elements=552950300,owner_growth_bytes=0,existing_union_policy_unchanged=True,
             existing_sm12_rules_unchanged=True,models=0,accuracy_claim=False,
             limitations=['Rank35 geometry-only forecast; no seven/32-network coverage claim.',
                          'Warm prices do not include new setup aliases or cold first-use views.',
                          'Floating-point association changes require paired native evaluation.'])
for path,body in ((D/'fragment_shared_stack_v1.py',fragment),(out,source),(D/'SHARED_STACK_BUILD_v1.json',json.dumps(receipt,indent=2)+'\n')):
 with path.open('x')as f:f.write(body)
print(json.dumps({k:v for k,v in receipt.items()if k not in ('plans','families')}))
