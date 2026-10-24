"""Fixed-table complete-owner pricing from passive chronology; stdlib only."""
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;H2=HERE.parent
PLAN=H2/'teammate_s32_rank15_dispatch_v1/result.json'
PLAN_SHA='a771c76ddf8fc7de30c13c7d9b5fa65c6a28826ba1210b52292959fa2c674eac'


def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def read(p):return json.loads(p.read_text())


def check(p):
    assert pin(Path(p['path']))==p


def binary(d,m,k,n):
    g=Fraction(7,4)**d
    return tuple(map(int,((g+Fraction(d+3,4))*m*k,(g+Fraction(d+3,4))*k*n,
                         (2*g-Fraction(3,4)+Fraction(7*(d-1),16))*m*n,
                         7**d*(m//2**d)*(n//2**d)*(2*(k//2**d)-1))))


def assess_row(receipt,table):
    check(receipt['trace']);events=[]
    with Path(receipt['trace']['path']).open() as f:
        for line in f:
            e=json.loads(line);assert e['event']==len(events);events.append(e)
    assert len(events)==receipt['observer']['events']<=100000
    assert events[0]['kind']=='start' and events[-1]['kind']=='finish'
    roots={};products=defaultdict(list);ks_products=defaultdict(list);stack=[]
    for e in events:
        if e['kind']=='begin':
            stack.append(e['call'])
            if e['is_root']:roots[e['call']]=dict(begin=e)
        elif e['kind']=='end':
            assert e['ok'] and stack.pop()==e['call']
            if e['call'] in roots:roots[e['call']]['end']=e
        elif e['kind']=='product':
            products[e['root']].append(e)
            if e['ks'] is not None:ks_products[e['ks']].append(e)
    assert not stack
    selected=set();proposals=[];exclusions=[]
    for rid,r in roots.items():
        b=r['begin'];p=products[rid];end=r['end'];charges=end['charges']
        assert len(p)==charges['products']
        matches=[x for x in p if x['supported'] and tuple([x['depth']]+x['shape']) in table]
        if not matches:continue
        reason=[]
        if b['grant']:reason.append('active_W5_grant')
        if b['blocked']:reason.append('internal_kernel_ancestor')
        if charges['unsupported_products']:reason.append('unaccounted_product_family')
        if any(x is None or len(x['shape'])!=2 or x['dtype']!='float32' for x in b['inputs'][:2]):
            reason.append('not_two_FP32_matrices')
        if reason:
            exclusions.append(dict(root=rid,reasons=reason,positive_products=len(matches)));continue
        proposed=0;old_independent=0
        for x in p:
            parts=binary(x['depth'],*x['shape']);assert parts[-1]==x['paid']
            old=sum(parts);old_independent+=old
            q=table.get(tuple([x['depth']]+x['shape']))
            proposed+=old if q is None else q['flops']
        sums=0
        for kid in {x['ks'] for x in p if x['ks'] is not None}:
            group=ks_products[kid]
            assert all(x['root']==rid and x['supported'] for x in group)
            mn={(x['shape'][0],x['shape'][2]) for x in group};assert len(mn)==1
            m,n=next(iter(mn));sums+=(len(group)-1)*m*n
        proposed+=sums
        selected.add(rid)
        proposals.append(dict(root=rid,layer=b['layer'],products=len(p),HK_products=len(matches),
            actual_warm=charges['warm'],original_cold=charges['cold'],unchanged_other=charges['other'],
            old_independent=old_independent,proposed_warm=proposed,KS_output_add=sums,
            warm_saving_before_future_hits=charges['warm']-proposed))
    # Actual marker identity is recorded via weak lifetime tokens. Counterfactual
    # state is just validity relative to that actual marker, never array values.
    actual=None;valid=False;active=None;extra=[];a_pending=[];verified=Counter()
    def sync(marker):
        nonlocal actual,valid
        if marker is None:actual=None;valid=False
        else:assert marker==actual,'unobserved_nonzero_marker_assignment'
    for e in events:
        kind=e['kind']
        if kind=='start':actual=e['marker'];valid=actual is not None
        elif kind=='begin':
            sync(e['marker'])
            if e['is_root'] and e['call'] in selected:
                assert active is None;active=e['call'];valid=False
        elif kind=='end':
            sync(e['marker'])
            if e['call'] in selected:
                assert active==e['call'];active=None;valid=False
        elif kind=='a_begin':
            sync(e['marker']);a_pending.append(e)
            assert e['actual_hit']==(actual==[e['input'],e['frame']])
        elif kind=='a_end':
            a=a_pending.pop();assert (a['input'],a['frame'],a['root'])==(e['input'],e['frame'],e['root'])
            d,m,k=e['frame'];fee=binary(d,m,k,2**d)[0]
            if e['warm'] is not None:
                assert e['warm']==(0 if e['actual_hit'] else fee),'A_forward_tariff_mismatch'
                verified['hits' if e['actual_hit'] else 'computed']+=1
            if e['actual_hit'] and active is None and not valid:
                extra.append(dict(root=e['root'],event=e['event'],input=e['input'],frame=e['frame'],flops=fee))
            actual=e['marker'];valid=active is None
        elif kind=='finish':sync(e['marker'])
    assert active is None and not a_pending
    before=sum(r['warm_saving_before_future_hits'] for r in proposals)
    lost=sum(r['flops'] for r in extra)
    return dict(rank=receipt['rank'],flops=receipt['flops'],selected_complete_owners=len(selected),
        proposals=proposals,excluded_positive_owners=exclusions,verified_A_forward_events=dict(verified),
        lost_later_unselected_actual_hits=extra,lost_hit_flops=lost,
        warm_saving_before_lost_hits=before,conditional_warm_saving=before-lost,
        conditional_warm_pct=100*(before-lost)/receipt['flops'],
        cold_bind_transfer_assumed=False,UBC_credit_added=0,original_all16_and_bill_exact=True)


def main():
    assert not (HERE/'assessment.json').exists()
    assert pin(PLAN)['sha256']==PLAN_SHA
    done=read(HERE/'complete.json');guard=read(HERE/'guard.json')
    assert done['passed'] and done['ranks']==[35,100] and not done['targets_accessed']
    assert guard['completed'] and guard['exit_code']==0 and guard['error'] is None
    assert guard['whole_group_retired'] and not guard['remaining_rows']
    native=H2.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    assert guard['command']==[str(native),'-B',str(HERE/'run.py')]
    assert 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] <= 7500000000
    assert 0 < guard['elapsed_s'] <= 180 and guard['samples'] > 0
    assert guard['max_swap_growth_bytes']==0
    for k in ('source','seal','data_refs','setup'):check(done[k])
    seal=read(Path(done['seal']['path']))
    for p in seal['sources']:check(p)
    table={}
    for r in read(PLAN)['one_algorithm_shape_dispatch']['positive']:
        key=tuple([r['depth']]+r['shape']);q=r['alternative']
        assert key not in table or table[key]==q;table[key]=q
    assert len(table)==54
    rows=[]
    for p in done['rows']:
        check(p);r=read(Path(p['path']))
        assert r['passed'] and r['all16_bytes_exact'] and r['full_bill_exact'] and r['observer']['patches_restored']
        rows.append(assess_row(r,table))
    assert [r['rank'] for r in rows]==[35,100]
    out=dict(passed=True,status='UNCHANGED_CHRONOLOGY_AND_CONDITIONAL_OWNER_PRICE',rows=rows,
        mean_conditional_warm_pct=sum(r['conditional_warm_pct'] for r in rows)/2,
        source=done['source'],guard=pin(HERE/'guard.json'),complete=pin(HERE/'complete.json'),
        fixed_dispatch=pin(PLAN),assessor=pin(Path(__file__)),targets_accessed=False,
        candidate_built=False,HK_runtime_qualified=False,HK_memory_qualified=False,
        limitations=['Frozen54shape/depth table only; unknown or W5-granted owners stay original.',
            'Counterfactual pricing preserves unchanged outer padding/copies and debits complete-owner warm kernels once.',
            'Complete-owner replacement must reproduce the recorded schedule; local TA/TC may not survive arbitrary inner callbacks.',
            'No cold/setup/UBC credit or new-operand floating-point downstream cost changes are inferred.',
            'Trace proves native source chronology and lost-hit tariff, not a implemented HK kernel or its array disjointness.'])
    with (HERE/'assessment.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,mean_conditional_warm_pct=out['mean_conditional_warm_pct'])))


if __name__=='__main__':main()
