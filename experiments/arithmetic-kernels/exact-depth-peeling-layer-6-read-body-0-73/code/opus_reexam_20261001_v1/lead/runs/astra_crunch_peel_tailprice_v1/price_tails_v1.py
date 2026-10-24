"""Metadata-only fixed fringe alternatives; original selected cutsets/body intact."""
from pathlib import Path
import collections,gc,json,resource,time

HERE=Path(__file__).resolve().parent
R=HERE.parent
D=R/'astra_crunch_compute_v1/peel_sm40_v1'
ns={'__file__':str(D/'price_peel_v1.py'),'__name__':'tail_metadata'}
exec((D/'price_peel_v1.py').read_text().split('push=json.loads')[0],ns)
M,fn,fs,arr,measure=(ns[x] for x in ('M','fn','fs','arr','measure'))
old=json.loads((D/'TARIFF_v1.json').read_text())
selected=[x for x in old['rows'] if x['best'] and x['best']['saving']>0]
shapes=sorted({tuple(t['shape']) for x in selected for t in x['best']['tails']})
depths=[]
forward_a=M._ipt_forward_a_ab
forward_b=M._ipt_forward_b_ab
def fa(a,L,m,k):
    depths.append((int(L),int(m),int(k)))
    return forward_a(a,L,m,k)
def fb(b,L,k,n):
    depths.append((int(L),int(k),int(n)))
    return forward_b(b,L,k,n)
M._ipt_forward_a_ab=fa
M._ipt_forward_b_ab=fb


def shallow(a,b,L):
    # Reuse original exact alt-basis tree implementation, bypassing its optional
    # L5->L6 promotion. This creates no new kernel and changes no global flags.
    M._TR_LAST[0]=None  # The preceding Sm40 body overwrites the shared arena.
    m,k,n=int(a.shape[0]),int(a.shape[1]),int(b.shape[1])
    u=1<<L
    mm,kk,nn=((x+u-1)//u*u for x in (m,k,n))
    assert M._ipt_fits(L,mm,kk,nn)
    A=a;B=b
    if (m,k)!=(mm,kk):
        A=fn.zeros((mm,kk),dtype=fn.float32);fn.copyto(A[:m,:k],a)
    if (k,n)!=(kk,nn):
        B=fn.zeros((kk,nn),dtype=fn.float32);fn.copyto(B[:k,:n],b)
    out=arr((mm,nn))
    M._P6_OLD_IPAB(A,B,out,L)
    if (m,n)!=(mm,nn):
        return fn.copy(out[:m,:n])
    return out


t0=time.monotonic();tails={}
for i,shape in enumerate(shapes):
    m,k,n=shape
    with fs.BudgetContext(flop_budget=2**60,quiet=True):
        a,b=arr((m,k)),arr((k,n))
    measure(lambda:M._mm(a,b));depths.clear()
    parent=measure(lambda:M._mm(a,b));used=list(depths)
    L=max([d[0] for d in used],default=0)-1
    v=dict(shape=shape,parent=parent,parent_forward_depths=used,lower_depth=L)
    u=1<<L if L>=2 else 0
    if u and M._ipt_fits(L,*[(x+u-1)//u*u for x in shape]):
        measure(lambda:shallow(a,b,L));v['shallower']=measure(lambda:shallow(a,b,L))
    if min(shape)<=64:
        measure(lambda:fn.matmul(a,b));v['dense']=measure(lambda:fn.matmul(a,b))
    tails[shape]=v
    M._BH_HOLD.clear();gc.collect()
    if i%30==0:print(json.dumps(dict(stage='tails',i=i,total=len(shapes),seconds=time.monotonic()-t0)),flush=True)


def choose(v,policy):
    choices=['parent']
    if policy in ('shallower','shallower_or_dense','bill_safe') and 'shallower'in v:choices.append('shallower')
    if policy in ('dense_skinny','shallower_or_dense','bill_safe') and 'dense'in v:choices.append('dense')
    if policy=='bill_safe':
        choices=[x for x in choices if v[x]['flops']<=v['parent']['flops']]
    # Fixed request-priority policy; original parent remains an option.
    return min(choices,key=lambda x:(v[x]['requests'],v[x]['flops']))


policies=('shallower','dense_skinny','shallower_or_dense','bill_safe')
by_old={(x['kind'],tuple(x['shape'])):x for x in selected}
results={}
for policy in policies:
    rows=[]
    for x in selected:
        b=x['best'];df=dq=0;plans=[]
        for tail in b['tails']:
            shape=tuple(tail['shape']);v=tails[shape];method=choose(v,policy)
            df+=v[method]['flops']-v['parent']['flops'];dq+=v[method]['requests']-v['parent']['requests']
            plans.append(dict(kind=tail['kind'],shape=shape,method=method,lower_depth=v['lower_depth']))
        rows.append(dict(kind=x['kind'],shape=x['shape'],cuts=b['cuts'],scheme=b['scheme'],tail_plans=plans,
                         flops=b['flops']+df,requests=b['requests']+dq,saving=b['saving']-df,request_delta=b['request_delta']+dq,
                         saving_lost=df,requests_removed=-dq))
    by={(x['kind'],tuple(x['shape'])):x for x in rows}
    coverage={}
    for kind,name in [('push','OUTER_COMBINED_TARIFF_v3.json'),('read','READ_TARIFF_v2.json')]:
        col=json.loads((R/'astra_crunch_compute_v1/bini_apa_v1'/name).read_text())
        cov=[]
        for c in col['coverage']:
            eligible=[by[kind,tuple(sh)] for sh in c['hits'] if (kind,tuple(sh))in by and by[kind,tuple(sh)]['saving']>0]
            denom=c['total'] if kind=='push' else c['bill']
            cov.append(dict(seed=c['seed'],denominator=denom,gain=sum(v['saving']for v in eligible),
                            request_delta=sum(v['request_delta']for v in eligible),calls=len(eligible),
                            methods=dict(collections.Counter(p['method']for v in eligible for p in v['tail_plans']))))
        coverage[kind]=cov
    read={x['seed']:x for x in coverage['read']}
    matched=[dict(seed=x['seed'],combined_pct=100*(x['gain']+read[x['seed']]['gain'])/x['denominator'],
                  request_delta=x['request_delta']+read[x['seed']]['request_delta'])for x in coverage['push']]
    summary=dict(mean_pct=sum(x['combined_pct']for x in matched)/len(matched),
                 mean_request_delta=sum(x['request_delta']for x in matched)/len(matched),
                 selected_shapes=sum(x['saving']>0 for x in rows))
    results[policy]=dict(summary=summary,rows=rows,coverage=coverage,matched8=matched)
    print(json.dumps(dict(policy=policy,**summary)),flush=True)

out=dict(status='METADATA_PRICING_FIXED_SELECTED_CUTSETS',
         source_sha=ns['ns']['p'].SOURCE_SHA,
         original_summary=old['summary'],tails=list(tails.values()),policies=results,
         seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         limitations=['Full original body/packing/copy/assembly tariff retained; each measured tail replaced additively.',
                     'No new cuts or previously losing shapes explored.',
                     'Requests are wrapper-operation proxy, not residual instructions or runtime.',
                     'Lower depth is one less than deepest forward transform used by current parent; unsupported arena geometries keep parent.',
                     'Dense considered only for min(m,k,n)<=64. Request-priority options keep parent when it has fewer calls.',
                     'Metadata arrays contain no numerical values; numeric parity/runtime not claimed.'])
(HERE/'TARIFF_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(seconds=out['seconds'],maxrss=out['maxrss'])),flush=True)
