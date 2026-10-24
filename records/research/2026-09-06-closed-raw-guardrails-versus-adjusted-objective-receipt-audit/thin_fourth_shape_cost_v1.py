"""Shape-only arithmetic counter for offline architecture pricing.

This does not invoke or replace flopscope, run an estimator, create wide
arrays, or claim measured candidate FLOPs. Counters execute the certified
query algebra with array shapes. Copy/index/broadcast/view bytes are not
arithmetic; memory/latency are separate unresolved gates.
"""
from collections import Counter
import math
import numpy as np


class Ledger:
    def __init__(self,mat_rate=2):
        self.mat_rate = mat_rate
        self.counts = Counter()
        self.group = 'unassigned'

    def add(self,kind,flops):
        self.counts[self.group+'/'+kind] += float(flops)

    def shape(self,*shape):
        return Shape(shape,self)

    @property
    def total(self):
        return sum(self.counts.values())


class Shape:
    __array_priority__ = 1000
    def __init__(self,shape,ledger):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)
        self.ledger = ledger

    def __len__(self):
        return self.shape[0]

    def __float__(self):
        assert self.shape == ()
        # Only shape-independent source balancing and q0 scalar factors
        # request float. No branch in the query depends on this value.
        return 1.

    def __iter__(self):
        for j in range(len(self)):
            yield self[j]

    def __getitem__(self,key):
        if not isinstance(key,tuple):
            key = (key,)
        answer,axis = [],0
        for index in key:
            if index is None:
                answer.append(1)
            elif isinstance(index,slice):
                answer.append(len(range(*index.indices(self.shape[axis])))); axis += 1
            elif isinstance(index,(int,np.integer)):
                axis += 1
            else:
                raise TypeError((self.shape,key))
        answer += self.shape[axis:]
        return Shape(answer,self.ledger)

    def __setitem__(self,key,value):
        pass

    @property
    def T(self):
        return Shape(self.shape[::-1],self.ledger)

    def copy(self):
        return Shape(self.shape,self.ledger)

    clone = copy

    def _point(self,other=0,cost=1):
        shape = np.broadcast_shapes(self.shape,other.shape if isinstance(other,Shape) else np.shape(other))
        self.ledger.add('f64_pointwise',2*cost*math.prod(shape))
        return Shape(shape,self.ledger)

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _point
    __truediv__ = __rtruediv__ = _point

    def __neg__(self):
        return self._point()

    def __pow__(self,power):
        return self._point(cost=16)

    def __matmul__(self,other):
        assert isinstance(other,Shape)
        a,b = self.shape,other.shape
        assert 1 <= len(a) <= 2 and 1 <= len(b) <= 2,(a,b)
        m,k = a if len(a) == 2 else (1,a[0])
        kb,n = b if len(b) == 2 else (b[0],1)
        assert k == kb,(a,b)
        self.ledger.add('gemm_or_dot',self.ledger.mat_rate*m*n*max(0,2*k-1))
        shape = (m,n) if len(a) == len(b) == 2 else ((m,) if len(a)==2 else ((n,) if len(b)==2 else ()))
        return Shape(shape,self.ledger)

    def sum(self,axis=None):
        shape = list(self.shape)
        if axis is None:
            shape = []
        else:
            del shape[axis]
        self.ledger.add('f64_reduction',2*max(0,self.size-math.prod(shape)))
        return Shape(shape,self.ledger)

    def diag(self):
        return Shape((min(self.shape),),self.ledger) if self.ndim==2 else Shape((self.size,self.size),self.ledger)


class Backend:
    def __init__(self,ledger):
        self.L = ledger
        self.linalg = self

    def zeros(self,shape):
        return Shape((shape,) if isinstance(shape,int) else shape,self.L)

    ones = zeros

    def zeros_like(self,x):
        return x.copy()

    ones_like = zeros_like

    def fill_diagonal(self,*args):
        pass

    def diag(self,x):
        return x.diag()

    def outer(self,x,y):
        return x[:,None]*y[None,:]

    def concatenate(self,values,axis=0):
        shape = list(values[0].shape)
        shape[axis] = sum(v.shape[axis] for v in values)
        return Shape(shape,self.L)

    def stack(self,values,axis=0):
        shape = list(values[0].shape)
        shape.insert(axis,len(values))
        return Shape(shape,self.L)

    def asarray(self,values):
        return Shape(np.shape(values),self.L)

    def norm(self,x):
        self.L.add('f64_norm',2*(2*x.size))
        return Shape((),self.L)
