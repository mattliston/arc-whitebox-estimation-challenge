"""Same fixed peel cuts/tails against exact S2tr; public metadata only."""
from pathlib import Path
import ast,collections,gc,hashlib,json,resource,time
HERE=Path(__file__).resolve().parent;R=HERE.parent
T=R/'astra_crunch_peel_tailprice_v1'
ns={'__file__':str(T/'price_tails_v1.py'),'__name__':'s2tr_metadata'}
exec((T/'price_tails_v1.py').read_text().split('t0=time.monotonic()')[0],ns)
P,fn,fs,arr,measure=(ns[k]for k in ('M','fn','fs','arr','measure'))
probe=ns['ns']['ns']['p']
pub_sha=probe.SOURCE_SHA
probe.SOURCE=R/'opus_s2tr_v1/candidate_S2tr_v1.py'
probe.SOURCE_SHA=hashlib.sha256(probe.SOURCE.read_bytes()).hexdigest()
S,_=probe.load();S._WINOGRAD=True;S._AB_ON=True;S._YR7_ACTIVE[0]=True
with fs.BudgetContext(flop_budget=2**60,quiet=True):
    arena=arr((S._PACKED_ELEMENTS,))
    S._SCRATCH=[arena,S._cp3_bind_arena_views(arena,S._PACKED_PUSH),S._cp3_bind_arena_views(arena,S._PACKED_READ),None,None,None,None]
    S._bind_push6(arena);S._ASTRA_SM40_ENV._SCRATCH=S._SCRATCH
    S._astra_sm40_bind_universal(S._ASTRA_SM40_ENV,arena)


def cost(call):
    S._BH_HOLD.clear();S._TR_LAST[0]=None
    return measure(call)


def shallow(a,b,L):
    S._TR_LAST[0]=None
    m,k,n=int(a.shape[0]),int(a.shape[1]),int(b.shape[1]);u=1<<L
    mm,kk,nn=((x+u-1)//u*u for x in (m,k,n));assert S._ipt_fits(L,mm,kk,nn)
    A=a;B=b
    if (m,k)!=(mm,kk):A=fn.zeros((mm,kk),dtype=fn.float32);fn.copyto(A[:m,:k],a)
    if (k,n)!=(kk,nn):B=fn.zeros((kk,nn),dtype=fn.float32);fn.copyto(B[:k,:n],b)
    out=arr((mm,nn));S._P6_OLD_IPAB(A,B,out,L)
    return fn.copy(out[:m,:n])if (m,n)!=(mm,nn)else out


fixed=json.loads((T/'DEPLOY_STRICT750_v1.json').read_text())
source=(R/'astra_crunch_compute_v1/peel_sm40_v1/price_peel_v1.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='make')
make_source=ast.get_source_segment(source,node).replace('def make(a,b,record,cuts):','def make_s2tr(a,b,record,cuts,tailplans):').replace('z=M._mm(A,B)','z=dispatch(A,B,tailplans[kind])')
def dispatch(a,b,plan):
    method=plan['method']
    if method=='parent':return S._mm(a,b)
    if method=='dense':return fn.matmul(a,b)
    return shallow(a,b,plan['lower_depth'])
make_ns=dict(ns['ns'],dispatch=dispatch)
exec(make_source,make_ns);make=make_ns['make_s2tr']
t0=time.monotonic();rows=[]
for i,spec in enumerate(fixed['rows']):
    kind=spec['kind'];shape=spec['shape'];m,k,n=shape[:3];rank=shape[3]if kind=='read'else 0;old=int(rank>0)
    with fs.BudgetContext(flop_budget=2**60,quiet=True):
        A=arr((m,k+rank));B=arr((k+rank,n));youngA=arr((m,k));youngB=arr((k,n));lam=arr((k,))
        oldA=arr((m,rank));oldB=arr((rank,n));ya=A[:,:k];yb=B[:k,:];oa=A[:,k:];ob=B[k:,:]
        full=arr((m,k+old));yr=arr((n,k));full_lam=arr((k+old,));inner=arr((m,rank));right=arr((n,rank))
        idx=fn.asarray(probe.np.arange(n,dtype='int64'));state=dict(columns=old,pre_basis=arr((1024,rank)))
        run,info=make(A,B,spec['body_record'],spec['cuts'],{p['kind']:p for p in spec['tail_plans']})
    S._cp_slr_inner_sq=lambda *args:inner
    S._xsg_rows=lambda *args:right
    baseline=(lambda:S._astra_sm40_read(full,yr,full_lam,idx,state))if kind=='read'else(lambda:S._mm(youngA,youngB))
    cost(baseline);base=cost(baseline)
    def candidate():
        if kind=='read':
            if m<n:fn.multiply(youngA,lam,out=ya);fn.copyto(yb,youngB)
            else:fn.copyto(ya,youngA);fn.multiply(youngB,lam[:,None],out=yb)
            if rank:fn.copyto(oa,oldA);fn.copyto(ob,oldB)
        return run()
    cost(candidate);price=cost(candidate)
    rows.append(dict(spec,s2tr_base=base,s2tr_price=price,s2tr_saving=base['flops']-price['flops'],
                     s2tr_request_delta=price['requests']-base['requests']))
    if i%20==0:print(json.dumps(dict(progress=i,total=len(fixed['rows']),seconds=time.monotonic()-t0)),flush=True)
    del baseline,candidate,run;P._BH_HOLD.clear();S._BH_HOLD.clear();gc.collect()
    (HERE/'PARTIAL_v1.json').write_text(json.dumps(rows)+'\n')

# Same empirical sequences and denominators as the original shape census.
# Push sequence positions are available independently; paired combined ordering
# is only approximated below and is explicitly not a new native trace.
pc=json.loads((R/'astra_crunch_compute_v1/bini_apa_v1/OUTER_COMBINED_TARIFF_v3.json').read_text())['coverage']
rc=json.loads((R/'astra_crunch_compute_v1/bini_apa_v1/READ_TARIFF_v2.json').read_text())['coverage']
lookup={(r['kind'],tuple(r['shape'])):r for r in rows}
def replay(events,budget=None):
    spent=0;gain=0;net=0;accepted=[];skipped=[]
    for kind,shape in events:
        r=lookup.get((kind,tuple(shape)))
        if r is None or r['s2tr_saving']<=0:continue
        charge=max(0,r['s2tr_request_delta'])
        if budget is not None and spent+charge>budget:
            skipped.append(dict(kind=kind,shape=shape,charge=charge));continue
        spent+=charge;net+=r['s2tr_request_delta'];gain+=r['s2tr_saving'];accepted.append(dict(kind=kind,shape=shape,charge=charge))
    return dict(gain=gain,request_delta=net,positive_calls_spent=spent,accepted=accepted,skipped=skipped)
coverage={}
for kind,collection in [('push',pc),('read',rc)]:
    coverage[kind]=[]
    for c in collection:
        den=c['total']if kind=='push'else c['bill'];events=[(kind,sh)for sh in c['hits']]
        raw=replay(events);cap=replay(events,750)
        coverage[kind].append(dict(seed=c['seed'],denominator=den,raw=dict(raw,pct=100*raw['gain']/den),capped750=dict(cap,pct=100*cap['gain']/den)))
readseed={c['seed']:c for c in rc};matched=[]
for p in pc:
    r=readseed[p['seed']];den=p['total']
    pp=[('push',s)for s in p['hits']];rr=[('read',s)for s in r['hits']]
    modes={}
    for label,events in [('push_then_read',pp+rr),('read_then_push',rr+pp)]:
        q=replay(events,750);modes[label]=dict(q,pct=100*q['gain']/den)
    raw=replay(pp+rr)
    matched.append(dict(seed=p['seed'],denominator=den,raw=dict(raw,pct=100*raw['gain']/den),capped750_order_checks=modes))
summary=dict(matched8_raw_mean_pct=sum(r['raw']['pct']for r in matched)/len(matched),
             matched8_raw_mean_requests=sum(r['raw']['request_delta']for r in matched)/len(matched),
             matched8_raw_max_requests=max(r['raw']['request_delta']for r in matched),
             read24_raw_max_requests=max(r['raw']['request_delta']for r in coverage['read']),
             read24_capped_mean_pct=sum(r['capped750']['pct']for r in coverage['read'])/len(rc),
             matched8_capped_order_mean_pct={mode:sum(r['capped750_order_checks'][mode]['pct']for r in matched)/len(matched)for mode in ('push_then_read','read_then_push')},
             shapes=len(rows),profitable_shapes=sum(r['s2tr_saving']>0 for r in rows))
out=dict(status='FIXED_STRICT750_TABLE_REPRICED_AGAINST_S2TR_PUBLIC_METADATA',s2tr_source_sha=probe.SOURCE_SHA,
         body_source_sha=pub_sha,rows=rows,summary=summary,coverage=coverage,matched8=matched,
         seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,payload_bytes=probe.PAYLOAD[0],
         limits=['Same frozen79shape cuts/tail methods. No reoptimization or new geometry.',
                 'Actual public full recurring tariff for S2tr baseline and candidate with S2tr parent tails; generic exact SMX body uses unchanged pubmax implementation.',
                 'Whole percentages use previous saved census denominators, not freshly measured full S2tr model bills.',
                 'Budget debits max(0, extra wrapper calls) and falls back exact on exhaustion; bounds net added calls by750 for any sequence of listed shapes.',
                 'All24 read sequences replayed; push/read combined order is only push-first/read-first bounds here, not actual interleaving.',
                 'Warm wrapper-operation proxy only; residual instructions/runtime/accuracy not measured.'])
(HERE/'TARIFF_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(summary=summary,seconds=out['seconds'],maxrss=out['maxrss'],payload_bytes=out['payload_bytes'])),flush=True)
