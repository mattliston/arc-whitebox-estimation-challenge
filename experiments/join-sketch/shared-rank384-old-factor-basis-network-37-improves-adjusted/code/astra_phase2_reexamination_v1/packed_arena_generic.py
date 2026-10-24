"""Generalized audited parity/lifetime arena layout for Winograd depth L.

Only metadata is computed here. Numerical kernels remain FlopScope calls.
Final-parity transforms coexist with previous-parity transforms; the leaf
then reuses dead previous/dependency storage, and unwind reuses dead inputs.
"""
from fractions import Fraction
from math import prod

def layout(m,k,n,levels):
    assert all(v%2**levels==0 for v in (m,k,n))
    q=Fraction(7,4);a=m*k;b=k*n;c=m*n
    def exact(x):
        assert x.denominator==1
        return x.numerator
    last_a=exact(q**levels*a);last_b=exact(q**levels*b)
    prev_a=exact(q**(levels-1)*a);prev_b=exact(q**(levels-1)*b)
    dep=exact(q**(levels-1)*max(a,b));leaf=exact(q**levels*c);product=exact(q**(levels-1)*c)
    intervals={};cursor=0
    for name,size in (("last_a",last_a),("last_b",last_b),("prev_a",prev_a),("prev_b",prev_b),("dep",dep)):
        intervals[name]=(cursor,cursor+size);cursor+=size
    final_stop=last_a+last_b;leaf_start=cursor-leaf
    assert leaf<=prev_a+prev_b+dep and final_stop<=leaf_start and 2*product<=leaf_start
    intervals.update(leaf=(leaf_start,cursor),product0=(0,product),u=(product,2*product),product1=(leaf_start,leaf_start+product))
    specs=[]
    def view(owner,shape):
        start,limit=intervals[owner];stop=start+prod(shape)
        assert stop<=limit,(owner,shape,stop,limit)
        specs.append((start,stop,tuple(shape)))
    for l in range(1,levels+1):
        parity="last" if l%2==levels%2 else "prev"
        view(parity+"_a",(7,)*l+(m//2**l,k//2**l))
        view(parity+"_b",(7,)*l+(k//2**l,n//2**l))
    for x,y in ((m,k),(k,n)):
        for l in range(1,levels+1):view("dep",(4,)+(7,)*(l-1)+(x//2**l,y//2**l))
    view("leaf",(7,)*levels+(m//2**levels,n//2**levels))
    for unwind,l in enumerate(range(levels-1,0,-1)):
        view("product0" if unwind%2==0 else "product1",(7,)*l+(m//2**l,n//2**l))
    for l in range(levels,0,-1):view("u",(3,)+(7,)*(l-1)+(m//2**l,n//2**l))
    assert len(specs)==6*levels
    return cursor,tuple(specs),intervals
