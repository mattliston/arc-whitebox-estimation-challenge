"""Metadata-only exact primitive price of the existing D7/D6 read dispatch."""
import importlib.util,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
s=importlib.util.spec_from_file_location('virtual_tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py'
p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest()
M,_=p.load();fnp=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
    arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32)
    M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None]
    M._bind_push6(arena)
M._YR7_ACTIVE[0]=True;M._WINOGRAD=True;M._AB_ON=True
M._YR7_STATS=dict(calls=0,chunks=0,shapes=[],routed_shapes=[],old_fused=0,max_arena_elements=0)
M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
from collections import Counter
certs=list((O/'lead/runs/astra_read_tail_rank2_native_v1').glob('run_rtt2_science_*_s*_v1/native_*_ON.json'))
counts=Counter();pernet={};bills={}
for cp in certs:
    data=json.loads(cp.read_text());row=data['rows'][0];seed=row['seed'];bills[seed]=row['flops'];ss=[]
    for x in row['stats']['_RTT_STATS']['reads']:
        if x['forced_d7']:
            key=tuple(x[k]for k in ('m','k','c15','c2','old_rank'));assert key[-1]==0
            counts[key]+=1;ss.append(key)
    pernet[seed]=ss
assert len(pernet)==24
from flopscope.numpy import ndarray
counter=Counter();orig_get=ndarray.__getitem__;orig_T=ndarray.T
# Basic-index and .T views are one RPC each in numpy-free RemoteArray.
def counted_get(self,key):
    counter['getitem']+=1
    return orig_get(self,key)
ndarray.__getitem__=counted_get
def counted_T(self):
    counter['T']+=1
    return orig_T.__get__(self,type(self))
ndarray.T=property(counted_T)
cert=dict(capture_metadata=[dict(m=a,young_k=b,c15=c,c2=d,old_rank=r,old_columns=0,layer=None)for a,b,c,d,r in counts])
rows=[];fixed=M._YR7_SHAPES
for item in cert['capture_metadata']:
    m,k,n,r=item['m'],item['young_k'],item['c15'],item['old_rank'];c2=item['c2'];old=item['old_columns']
    key=(m,k,c2,r);new=(m,k,n,r)
    if n==c2 or not (key in fixed or ((m,k,c2) in M._R7_SHAPES and not old)):continue
    with fs.BudgetContext(flop_budget=2**60,quiet=True):
        a=fnp.empty((m,old+k),dtype=fnp.float32);right=fnp.empty((n,k),dtype=fnp.float32);lam=fnp.empty((old+k,),dtype=fnp.float32)
        idx=fnp.asarray(p.np.arange(n,dtype='int64'));inner=fnp.empty((m,r),dtype=fnp.float32);oldright=fnp.empty((n,r),dtype=fnp.float32)
        state=dict(columns=old,pre_basis=fnp.empty((1024,r),dtype=fnp.float32))
    # Common inner and row gather are excluded from both routes, unchanged by dispatch.
    M._cp_slr_inner_sq=lambda *args:inner;M._xsg_rows=lambda *args:oldright
    prices={}
    for name,shapes in [('forced_d7',fixed|frozenset((new,))),('existing_parent_route',fixed)]:
        M._YR7_SHAPES=shapes;M._TR_LAST[0]=None
        with fs.BudgetContext(flop_budget=2**60,quiet=True):M._yr7_read(a,right,lam,idx,state)
        counter.clear();p.CALLS.clear()
        with fs.BudgetContext(flop_budget=2**60,quiet=True) as ctx:answer=M._yr7_read(a,right,lam,idx,state)
        prices[name]=dict(flops=int(ctx.flops_used),counted_requests=len(ctx._op_log),backend_calls=dict(p.CALLS),free_views=dict(counter),estimated_rpc=sum(p.CALLS.values())+sum(counter.values()),shape=list(answer.shape))
        M._BH_HOLD.clear();M._TR_LAST[0]=None
    extra=prices['existing_parent_route']['flops']-prices['forced_d7']['flops']
    saved=prices['forced_d7']['counted_requests']-prices['existing_parent_route']['counted_requests']
    row=dict(layer=item['layer'],old_key=key,prefix_key=new,prices=prices,extra_flops=extra,saved_counted_requests=saved)
    row['saved_estimated_rpc']=prices['forced_d7']['estimated_rpc']-prices['existing_parent_route']['estimated_rpc'];rows.append(row);print(json.dumps(row),flush=True)
bykey={tuple(r['prefix_key'][:3]+[r['old_key'][2],r['old_key'][3]]):r for r in rows}
per={str(seed):dict(extra_flops=sum(bykey[k]['extra_flops']for k in ss),extra_bill_pct=100*sum(bykey[k]['extra_flops']for k in ss)/bills[seed],saved_estimated_rpc=sum(bykey[k]['saved_estimated_rpc']for k in ss),saved_charged_ops=sum(bykey[k]['saved_counted_requests']for k in ss))for seed,ss in pernet.items()}
out=dict(rows=rows,per_network=per,shape_counts=[dict(shape=k,count=v)for k,v in counts.items()],scope='Warm shape-only FNP charge plus explicit local getitem/T view RPC proxy; not full boundary RPC certification. Includes all forced D7 current24 shape routes. Ordinary35/100 shapes unavailable. No models/kernels; all old_rank zero.',numerical_payload_bytes=p.PAYLOAD[0])
out['summary']={k:dict(mean=sum(x[k]for x in per.values())/24,min=min(x[k]for x in per.values()),max=max(x[k]for x in per.values()))for k in ('extra_flops','extra_bill_pct','saved_estimated_rpc','saved_charged_ops')}
with(D/'CURRENT24_NO_FORCED_D7_TARIFF_v1.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out['summary']))
