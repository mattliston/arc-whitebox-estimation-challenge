"""Metadata-only exact request pricing of the fixed executable pilot.

No ndarray, native estimator, BLAS, or floating numerical work is executed.
"""
from collections import Counter
import json
from math import prod
from pathlib import Path
from kernel import CASES, HERE, Hybrid, binary_environment, region


class Symbol:
    dtype='float32'
    def __init__(self,shape,meter): self.shape=tuple(shape);self.meter=meter
    def __getitem__(self,key):
        if not isinstance(key,tuple): key=(key,)
        if Ellipsis in key:
            i=key.index(Ellipsis)
            key=key[:i]+(slice(None),)*(len(self.shape)-len(key)+1+sum(x is None for x in key))+key[i+1:]
        key=key+(slice(None),)*(len(self.shape)-sum(x is not None for x in key))
        shape=[];axis=0
        for s in key:
            if s is None: shape.append(1);continue
            size=self.shape[axis];axis+=1
            if isinstance(s,int): assert -size<=s<size
            else: shape.append(len(range(*s.indices(size))))
        assert axis==len(self.shape)
        return Symbol(shape,self.meter)
    def reshape(self,*shape):
        return self.meter.reshape(self,shape[0] if len(shape)==1 and isinstance(shape[0],(tuple,list)) else shape)


class Meter:
    float32='float32'
    def __init__(self): self.reset()
    def reset(self): self.cost=Counter();self.calls=Counter()
    def charge(self,name,count): self.cost[name]+=int(count);self.calls[name]+=1
    def empty(self,shape,dtype=None): return Symbol((shape,) if isinstance(shape,int) else shape,self)
    def zeros(self,shape,dtype=None): self.charge('zeros',0);return self.empty(shape)
    def reshape(self,a,shape):
        assert prod(shape)==prod(a.shape),(a.shape,shape)
        self.charge('reshape',prod(shape));return Symbol(shape,self)
    def transpose(self,a,axes):
        assert sorted(axes)==list(range(len(a.shape)))
        return Symbol(tuple(a.shape[i] for i in axes),self)
    def moveaxis(self,a,source,destination):
        axes=list(range(len(a.shape))); item=axes.pop(source);axes.insert(destination,item)
        return self.transpose(a,axes)
    def unstack(self,a,axis=0):
        return tuple(a[(slice(None),)*axis+(i,)] for i in range(a.shape[axis]))
    def copyto(self,out,a):
        assert out.shape==a.shape,(out.shape,a.shape)
        self.charge('copyto',prod(out.shape))
    def point(self,name,a,b,out):
        for arg in (a,b):
            if isinstance(arg,Symbol):
                assert len(arg.shape)<=len(out.shape)
                assert all(x in (1,y) for x,y in zip(reversed(arg.shape),reversed(out.shape))),(arg.shape,out.shape)
        self.charge(name,prod(out.shape));return out
    def add(self,a,b,out): return self.point('add',a,b,out)
    def subtract(self,a,b,out): return self.point('subtract',a,b,out)
    def multiply(self,a,b,out): return self.point('multiply',a,b,out)
    def matmul(self,a,b,out):
        assert a.shape[:-2]==b.shape[:-2]==out.shape[:-2]
        m,k=a.shape[-2:];k2,n=b.shape[-2:]
        assert k==k2 and out.shape[-2:]==(m,n)
        self.charge('matmul',prod(a.shape[:-2])*m*n*(2*k-1));return out
    def record(self): return dict(flops=sum(self.cost.values()),operation_flops=dict(self.cost),operation_calls=dict(self.calls))


def main():
    meter=Meter()
    shared_size=max(sum(region(c['shape'])) for c in CASES)
    shared_size=max(shared_size,max(sum(region(c['child'],40)) for c in CASES))
    shared=meter.empty(shared_size)
    result=dict(metadata_only=True,shared_arena_elements=shared_size,shared_arena_bytes=4*shared_size,cases=[])
    for case in CASES:
        m,k,n=case['shape']; A=meter.empty((m,k));B=meter.empty((k,n));out=meter.empty((m,n))
        base=binary_environment(meter,shared,case['shape'],1)
        hybrid=Hybrid(meter,shared,case)
        row=dict(case=case,records=[])
        for repetition in ('cold','warm'):
            for kind,call in [('parent',lambda:base['_ipt_product_ab'](A,B,out,5)),('smirnov',lambda:hybrid(A,B,out))]:
                meter.reset();call()
                row['records'].append(dict(kind=kind,repetition=repetition,**meter.record()))
        row['temporary_high_water_elements']=hybrid.arena.high_water
        row['temporary_peak_live_elements']=hybrid.arena.peak_live
        row['child_leaf_shape']=[x//32 for x in case['child']]
        row['shared_arena_required_elements']=sum(region(case['child'],40))
        row['outer_workspace_precise_bytes']=hybrid.workspace_bytes-4*hybrid.arena.capacity+4*hybrid.arena.high_water
        result['cases'].append(row)
    (HERE/'static.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
