"""Verify full public metadata prices for fixed alternative fringe policies."""
from pathlib import Path
import ast,gc,json,resource,time,collections
HERE=Path(__file__).resolve().parent
namespace={'__file__':str(HERE/'price_tails_v1.py'),'__name__':'full_metadata'}
exec((HERE/'price_tails_v1.py').read_text().split('t0=time.monotonic()')[0],namespace)
M,fn,fs,arr,measure=(namespace[k]for k in ('M','fn','fs','arr','measure'))
shallow=namespace['shallow'];D=namespace['D'];old=namespace['old'];R=namespace['R']
priced=json.loads((HERE/'TARIFF_v1.json').read_text())
tail_by={tuple(v['shape']):v for v in priced['tails']}
select_source=(HERE/'price_tails_v1.py').read_text()
node=next(n for n in ast.parse(select_source).body if isinstance(n,ast.FunctionDef)and n.name=='choose')
exec(ast.get_source_segment(select_source,node),globals())
def dispatch(a,b,policy):
    v=tail_by[(int(a.shape[0]),int(a.shape[1]),int(b.shape[1]))]
    method=choose(v,policy)
    if method=='parent':return M._mm(a,b)
    if method=='dense':return fn.matmul(a,b)
    return shallow(a,b,v['lower_depth'])
peel_source=(D/'price_peel_v1.py').read_text()
node=next(n for n in ast.parse(peel_source).body if isinstance(n,ast.FunctionDef)and n.name=='make')
make_source=ast.get_source_segment(peel_source,node).replace('def make(a,b,record,cuts):','def make_alt(a,b,record,cuts,policy):').replace('z=M._mm(A,B)','z=dispatch(A,B,policy)')
namespace['dispatch']=dispatch
namespace['body']=namespace['ns']['body']
exec(make_source,namespace)
make=namespace['make_alt']
start=time.monotonic();results={}
for policy in ('shallower','dense_skinny','shallower_or_dense'):
    rows=[]
    for i,spec in enumerate(priced['policies'][policy]['rows']):
        if spec['saving']<=0:continue
        kind=spec['kind'];shape=spec['shape'];m,k,n=shape[:3];rank=shape[3]if kind=='read'else 0
        with fs.BudgetContext(flop_budget=2**60,quiet=True):
            A=arr((m,k+rank));B=arr((k+rank,n));youngA=arr((m,k));youngB=arr((k,n));lam=arr((k,))
            oldA=arr((m,rank));oldB=arr((rank,n));ya=A[:,:k];yb=B[:k,:];oa=A[:,k:];ob=B[k:,:]
            run,info=make(A,B,'signed_executable_1'if kind=='read'else'signed_executable_0',spec['cuts'],policy)
        def call():
            if kind=='read':
                if m<n:fn.multiply(youngA,lam,out=ya);fn.copyto(yb,youngB)
                else:fn.copyto(ya,youngA);fn.multiply(youngB,lam[:,None],out=yb)
                if rank:fn.copyto(oa,oldA);fn.copyto(ob,oldB)
            return run()
        measure(call);v=measure(call)
        rows.append(dict(kind=kind,shape=shape,**v,expected_flops=spec['flops'],expected_requests=spec['requests'],
                         flop_difference=v['flops']-spec['flops'],request_difference=v['requests']-spec['requests']))
        del call,run;M._BH_HOLD.clear();gc.collect()
    result=dict(rows=rows,max_abs_flop_difference=max(abs(x['flop_difference'])for x in rows),
                max_abs_request_difference=max(abs(x['request_difference'])for x in rows),
                summary=priced['policies'][policy]['summary'])
    assert result['max_abs_flop_difference']==0,result
    # Calls used here have identical bound tail operands to the standalone tariff;
    # this assertion also checks that transform-reuse state did not earn fake credit.
    assert result['max_abs_request_difference']==0,result
    results[policy]=result
    print(json.dumps(dict(policy=policy,rows=len(rows),**{k:v for k,v in result.items()if k!='rows'})),flush=True)
out=dict(status='FULL_PUBLIC_METADATA_TARIFF_VERIFIED',results=results,
         seconds=time.monotonic()-start,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         source_sha=priced['source_sha'],payload_bytes=namespace['ns']['ns']['p'].PAYLOAD[0],
         scope='All selected winning shapes for three fixed policies; actual full body, read packing, tail arithmetic, output copies, and inner add. No model or numeric execution.')
(HERE/'FULL_VERIFY_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(seconds=out['seconds'],maxrss=out['maxrss'],payload_bytes=out['payload_bytes'])),flush=True)
