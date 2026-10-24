"""Passive, bounded NumPy reductions; never mutate or retain the input operands."""
import math
import numpy as np
BLOCK_BYTES=1024*1024

def count_terms(left,right):
    a=np.asarray(left);b=np.asarray(right)
    assert a.ndim>=2 and b.ndim==a.ndim and a.dtype==b.dtype==np.float32
    assert a.shape[:-2]==b.shape[:-2] and a.shape[-1]==b.shape[-2]
    m,k,n=int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])
    assert m>0 and k>0 and n>0 and all(x>0 for x in a.shape[:-2])
    batch=math.prod(a.shape[:-2]);totals=dict(batches=batch,dense_terms=batch*m*k*n,
        dense_flops=batch*m*n*(2*k-1),nonzero_terms=0,retained_lower_flops=0,
        zero_operand_batches=0,nonzero_left_entries=0,nonzero_right_entries=0,max_operand_chunk_bytes=0)
    def accumulate(ac,bc):
        t=np.sum(ac*bc,axis=-1,dtype=np.int64)
        totals['nonzero_terms']+=int(np.sum(t,dtype=np.int64))
        totals['retained_lower_flops']+=int(np.sum(t+np.maximum(t-m*n,0),dtype=np.int64))
        sa=np.sum(ac,axis=-1,dtype=np.int64);sb=np.sum(bc,axis=-1,dtype=np.int64)
        totals['zero_operand_batches']+=int(np.count_nonzero((sa==0)|(sb==0)))
        totals['nonzero_left_entries']+=int(np.sum(sa,dtype=np.int64))
        totals['nonzero_right_entries']+=int(np.sum(sb,dtype=np.int64))
    def checked(x,axis):
        assert x.nbytes<=BLOCK_BYTES and np.isfinite(x).all()
        totals['max_operand_chunk_bytes']=max(totals['max_operand_chunk_bytes'],int(x.nbytes))
        return np.count_nonzero(x,axis=axis)
    if a.flags.c_contiguous and b.flags.c_contiguous and max(m*k,k*n)*4<=BLOCK_BYTES:
        aa=a.reshape(batch,m,k);bb=b.reshape(batch,k,n)
        assert np.shares_memory(a,aa) and np.shares_memory(b,bb)
        step=max(1,BLOCK_BYTES//(4*max(m*k,k*n)))
        for i in range(0,batch,step):
            accumulate(checked(aa[i:i+step],-2),checked(bb[i:i+step],-1))
    else:
        assert 4*k<=BLOCK_BYTES
        for index in np.ndindex(a.shape[:-2]):
            aa=a[index];bb=b[index]
            ac=np.zeros(k,dtype=np.int64);bc=np.zeros(k,dtype=np.int64)
            step=max(1,BLOCK_BYTES//(4*k))
            for i in range(0,m,step):ac+=checked(aa[i:i+step,:],0)
            for j in range(0,n,step):bc+=checked(bb[:,j:j+step],1)
            accumulate(ac[None,:],bc[None,:])
    assert 0<=totals['nonzero_terms']<=totals['dense_terms']
    assert totals['nonzero_terms']<=totals['retained_lower_flops']<=totals['dense_flops']
    return dict(left_shape=list(a.shape),right_shape=list(b.shape),dtype=a.dtype.str,**totals)
