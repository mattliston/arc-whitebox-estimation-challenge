import hashlib,resource
import numpy as np
from common import *
from identity import fingerprint,signature,key,group_records,BLOCK_BYTES
seal_check();a=np.arange(60,dtype=np.float32).reshape(10,6);fixtures=[a,np.asfortranarray(a),a.T,a[::2,::2],a[::-1],a[:,::-1],a[:0],a[:,:0],np.array([[0.,-0.]],np.float32)]
checks=[]
for i,x in enumerate(fixtures):
    f=fingerprint(x);expected=hashlib.sha256(x.tobytes(order='C')).hexdigest()
    assert f['sha256']==expected and f['bytes']==x.nbytes and f['max_block_bytes']<=BLOCK_BYTES
    assert f['all_zero']==bool(np.all(x==0))
    checks.append(dict(case=i,shape=list(x.shape),canonical_exact=True))
assert fingerprint(a)['sha256']==fingerprint(np.asfortranarray(a))['sha256']
b=a.copy();b[0,0]=1;assert signature(fingerprint(a))!=signature(fingerprint(b))
nonfinite=0
for v in (float('nan'),float('inf'),float('-inf')):
    try:fingerprint(np.array([[v]],np.float32))
    except AssertionError:nonfinite+=1
assert nonfinite==3
A=np.arange(12,dtype=np.float32).reshape(3,4);B=np.arange(8,dtype=np.float32).reshape(4,2)
pairs=[(A,B),(A.copy(),B.copy()),(B.T,A.T),(A+1,B),(np.zeros_like(A),B),(A,np.zeros_like(B)),(A+1,B)]
rows=[]
for i,(left,right) in enumerate(pairs):
    result=left@right;fields={k:fingerprint(v) for k,v in dict(a=left,b=right,at=left.T,bt=right.T,c=result,ct=result.T).items()}
    rows.append(dict(index=i,flops=100+i,**fields))
original=[(x.copy(),y.copy()) for x,y in pairs]
exact=group_records(rows);extended=group_records(rows,True)
assert exact['duplicate_indices']==[1,6] and extended['duplicate_indices']==[1,2,6]
assert exact['zero_indices']==extended['zero_indices']==[4,5]
assert exact['optimistic_union_removed_flops']==101+104+105+106
assert extended['optimistic_union_removed_flops']==101+102+104+105+106
assert extended['hypothetical_retained_peak_bytes']==24
assert all(x['output_fingerprint_equal'] for x in extended['duplicates'])
# Independent pairwise equality, with no hashing or grouping helper.
for transpose,result in ((False,exact),(True,extended)):
    duplicates=[]
    for i,(x,y) in enumerate(pairs):
        for j,(u,v) in enumerate(pairs[:i]):
            direct=x.shape==u.shape and y.shape==v.shape and np.array_equal(x,u) and np.array_equal(y,v)
            reversed_pair=transpose and x.shape==v.T.shape and y.shape==u.T.shape and np.array_equal(x,v.T) and np.array_equal(y,u.T)
            if direct or reversed_pair:duplicates.append(i);break
    assert duplicates==result['duplicate_indices']
assert all(np.array_equal(x,u) and np.array_equal(y,v) for (x,y),(u,v) in zip(pairs,original))
write(HERE/'unit.json',dict(passed=True,fingerprint_fixtures=checks,nonfinite_rejections=nonfinite,
    independent_pairwise_groups_exact=True,input_preserved=True,exact=exact,transpose=extended,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,fingerprint_fixtures=len(checks),grouping_cases=2),flush=True)
