import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget
from common import *
from counts import count_terms,BLOCK_BYTES
seal_check()
def explicit(a,b):
    terms=lower=actual=zero=0;m,k,n=a.shape[-2],a.shape[-1],b.shape[-1]
    for ix in np.ndindex(a.shape[:-2]):
        t=q=0
        for i in range(m):
            for j in range(n):
                v=sum(int(a[ix+(i,h)]!=0 and b[ix+(h,j)]!=0) for h in range(k))
                t+=v;q+=int(v>0)
        terms+=t;lower+=t+max(t-m*n,0);actual+=2*t-q
        zero+=int(not np.any(a[ix]) or not np.any(b[ix]))
    return terms,lower,actual,zero
fixtures=[]
for m,k,n in ((2,3,4),(1,1,3),(4,2,1)):
    a=(np.arange(m*k).reshape(m,k)%3-1).astype(np.float32)
    b=(np.arange(k*n).reshape(k,n)%4-1).astype(np.float32)
    fixtures += [(a,b),(np.zeros_like(a),b),(a,np.ones_like(b)),(-a,-b)]
fixtures += [(np.eye(4,dtype=np.float32),np.eye(4,dtype=np.float32)),
 (np.array([[1,-0.,0],[0,0,2]],np.float32),np.array([[0,0],[3,4],[0,0]],np.float32))]
a=(np.arange(2*3*4).reshape(2,3,4)%3).astype(np.float32)
b=(np.arange(2*4*5).reshape(2,4,5)%2).astype(np.float32)
fixtures += [(a,b),(a[...,::-1],b[...,::-1,:]),(a.swapaxes(-1,-2),a),
 (np.stack([a,a]),np.stack([b,b]))]
rows=[]
for a,b in fixtures:
    before=(a.tobytes(),b.tobytes());a.flags.writeable=False;b.flags.writeable=False
    x=count_terms(a,b);e=explicit(a,b)
    assert (x['nonzero_terms'],x['retained_lower_flops'],x['zero_operand_batches'])==(e[0],e[1],e[3])
    assert x['retained_lower_flops']<=e[2]<=x['dense_flops']
    assert x==count_terms(a,b) and before==(a.tobytes(),b.tobytes())
    ap=a[...,::-1,:];bp=b[...,::-1];p=count_terms(ap,bp)
    assert all(p[k]==x[k] for k in ('nonzero_terms','retained_lower_flops','zero_operand_batches'))
    rows.append(dict(shape=[list(a.shape),list(b.shape)],terms=e[0],lower=e[1],exact_zero_skip=e[2],passed=True))
a=(np.arange(600*512).reshape(600,512)%5==0).astype(np.float32)
b=(np.arange(512*2).reshape(512,2)%7==0).astype(np.float32)
before=(a.tobytes(),b.tobytes());a.flags.writeable=False;b.flags.writeable=False
x=count_terms(a,b);ac=np.count_nonzero(a,axis=0);bc=np.count_nonzero(b,axis=1)
assert x['nonzero_terms']==sum(int(u)*int(v) for u,v in zip(ac,bc))
assert x['max_operand_chunk_bytes']<=BLOCK_BYTES<a.nbytes and before==(a.tobytes(),b.tobytes())
assert x==count_terms(a,b)
rejected=0
for a,b in [(np.ones((2,2),np.float64),np.ones((2,2),np.float32)),
 (np.ones((2,3),np.float32),np.ones((2,2),np.float32)),
 (np.full((2,2),np.nan,np.float32),np.ones((2,2),np.float32)),
 (np.ones((2,2),np.float32),np.full((2,2),np.inf,np.float32)),
 (np.ones((0,2),np.float32),np.ones((2,2),np.float32))]:
    try:count_terms(a,b)
    except AssertionError:rejected+=1
    else:raise AssertionError('Malformed fixture accepted')
assert rejected==5
original=_budget.BudgetContext._charge_op;seen=[]
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    fa=fnp.asarray(fixtures[0][0]);fb=fnp.asarray(fixtures[0][1])
with flops.BudgetContext(flop_budget=2**41,quiet=True) as base:ref=fnp.matmul(fa,fb)
def charged(ctx,*args,**kwargs):
    before=ctx._flops_used;answer=original(ctx,*args,**kwargs)
    if kwargs.get('op_name',args[0] if args else None)=='matmul':
        row=count_terms(fa,fb);assert row['dense_flops']==ctx._flops_used-before;seen.append(row)
    return answer
try:
    _budget.BudgetContext._charge_op=charged
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as obs:got=fnp.matmul(fa,fb)
finally:_budget.BudgetContext._charge_op=original
assert len(seen)==1 and base.summary_dict()['flops_used']==obs.summary_dict()['flops_used']==seen[0]['dense_flops']
assert np.asarray(ref).tobytes()==np.asarray(got).tobytes()
write(HERE/'unit.json',dict(passed=True,fixtures=rows,large_streaming=x,rejections=rejected,
    input_bytes_preserved=True,repeats_exact=True,actual_fnp_bill_and_output_exact=True,callback_restored=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,fixtures=len(rows),rejections=rejected),flush=True)
