"""One exact, fully padded, three-stage schedule; injected FNP or symbolic API."""
from math import prod

def orientation(shape):
    large=[i for i,d in enumerate(shape) if d>1024]
    if len(large)!=1 or min(shape)<64:return None
    return {2:'signed_executable_0',1:'signed_executable_1',0:'signed_executable_2'}[large[0]]

def normalized_shape(shape,partition):
    return tuple(((d+2*p**3-1)//(2*p**3))*(2*p**3) for d,p in zip(shape,partition))

class NormalizedProduct:
    def __init__(self,record,arena,api,block,prepare,linear,leaf):
        self.part=tuple(record['shape']);self.arena=arena;self.api=api
        self.Block=block;self.linear=linear;self.leaf=leaf
        self.circuits={key:prepare(record['circuits'][key]) for key in ('L','R','P')}
    def transform(self,a,rows,cols,key,consume):
        b,m,n=a.shape;assert m%rows==n%cols==0
        h,w=m//rows,n//cols
        inputs=[a.data[:,i*h:(i+1)*h,j*w:(j+1)*w] for i in range(rows) for j in range(cols)]
        out=self.arena.allocate((40*b,h,w));v=out.data.reshape(40,b,h,w)
        self.linear(self.circuits[key],inputs,[v[i] for i in range(40)],(b,h,w),self.arena)
        if consume:self.arena.release(a)
        return out
    def combine(self,products,out):
        p,_,q=self.part;b,m,n=out.shape;h,w=m//p,n//q
        inputs=products.data.reshape(40,b,h,w)
        outputs=[out.data[:,i*h:(i+1)*h,j*w:(j+1)*w] for i in range(p) for j in range(q)]
        self.linear(self.circuits['P'],[inputs[i] for i in range(40)],outputs,(b,h,w),self.arena)
    def leaves(self,a,b):
        batch,m,k=a.shape;n=b.shape[-1];assert m%2==k%2==n%2==0
        out=self.arena.allocate((batch,m,n))
        for start in range(0,batch,12800):
            stop=min(batch,start+12800);q=stop-start
            wa=self.arena.allocate((3,q,m//2,k//2));wb=self.arena.allocate((3,q,k//2,n//2));wp=self.arena.allocate((3,q,m//2,n//2))
            self.leaf(a.data[start:stop],b.data[start:stop],out.data[start:stop],wa.data,wb.data,wp.data)
            for v in (wa,wb,wp):self.arena.release(v)
        self.arena.release(a);self.arena.release(b)
        return out
    def descend(self,a,b,remaining):
        if not remaining:return self.leaves(a,b)
        batch,m,k=a.shape;n=b.shape[-1];p,q,r=self.part
        left=self.transform(a,p,q,'L',True);right=self.transform(b,q,r,'R',True)
        products=self.descend(left,right,remaining-1)
        out=self.arena.allocate((batch,m,n));self.combine(products,out);self.arena.release(products)
        return out
    def __call__(self,a,b):
        m,k=a.shape;n=b.shape[1];assert b.shape[0]==k
        M,K,N=normalized_shape((m,k,n),self.part);fnp=self.api
        self.padding_bytes=0
        if (M,K)!=(m,k):
            aa=fnp.zeros((M,K),dtype=fnp.float32);fnp.copyto(aa[:m,:k],a);self.padding_bytes+=4*M*K
        else:aa=a
        if (K,N)!=(k,n):
            bb=fnp.zeros((K,N),dtype=fnp.float32);fnp.copyto(bb[:k,:n],b);self.padding_bytes+=4*K*N
        else:bb=b
        out=fnp.empty((M,N),dtype=fnp.float32);self.output_bytes=4*M*N
        aa=self.Block(aa.reshape(1,M,K),(1,M,K));bb=self.Block(bb.reshape(1,K,N),(1,K,N));oo=self.Block(out.reshape(1,M,N),(1,M,N))
        p,q,r=self.part
        left=self.transform(aa,p,q,'L',False);right=self.transform(bb,q,r,'R',False)
        products=self.descend(left,right,2)
        self.combine(products,oo);self.arena.release(products)
        assert not self.arena.owned and self.arena.live==0
        return out[:m,:n]

def arithmetic_formula(shape,record):
    m,k,n=normalized_shape(shape,record['shape']);p,q,r=record['shape'];a,b,c=record['arithmetic']
    total=0;batch=1
    for _ in range(3):
        m//=p;k//=q;n//=r
        total+=batch*(a*m*k+b*k*n+c*m*n);batch*=40
    assert m%2==k%2==n%2==0
    x,y,z=m//2,k//2,n//2
    total+=batch*(7*(2*x*y*z-x*z)+4*x*y+4*y*z+7*x*z)
    return total
