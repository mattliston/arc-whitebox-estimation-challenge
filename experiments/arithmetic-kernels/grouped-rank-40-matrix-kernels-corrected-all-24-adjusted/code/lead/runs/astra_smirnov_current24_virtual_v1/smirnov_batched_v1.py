"""Inactive exact FNP-only rank40 outer circuit, current batched AB children.

No source import, model, setup or predict is run by this module. Caller supplies
the privately imported candidate module and a local certified circuit record.
"""
from collections import Counter
from fractions import Fraction
import math

def prepare(raw):
    nodes=raw['nodes'];outputs=[o['ref'] for o in raw['outputs']]
    assert all(Fraction(o['scale'])==1 for o in raw['outputs'])
    live=set()
    def visit(i):
        if i not in live:
            live.add(i)
            for a in nodes[i]['args']:visit(a)
    for i in outputs:visit(i)
    uses=Counter(a for i in live for a in nodes[i]['args'])
    destinations={}
    for i,ref in enumerate(outputs):destinations.setdefault(ref,[]).append(i)
    ops=[]
    for i in sorted(live):
        n=nodes[i];lit=n['literal']
        if n['op']=='scale':lit=float(Fraction(lit))
        ops.append((i,n['op'],tuple(n['args']),lit))
    return ops,uses,destinations

def linear(fnp,circuit,inputs,outputs,shape,stats):
    operations,uses,destinations=circuit;remaining=uses.copy();values={};owners=set()
    for ref,op,args,lit in operations:
        dest=destinations.get(ref,())
        if op=='input':value=inputs[lit]
        else:
            if dest:value=outputs[dest[0]]
            else:
                value=fnp.empty(shape,dtype=fnp.float32);owners.add(ref)
                stats['temp_live']+=math.prod(shape)*4
                stats['temp_peak']=max(stats['temp_peak'],stats['temp_live'])
            operands=[values[a]for a in args]
            if op=='add':fnp.add(operands[0],operands[1],out=value)
            elif op=='sub':fnp.subtract(operands[0],operands[1],out=value)
            elif op in ('scale','neg'):fnp.multiply(operands[0],-1 if op=='neg'else lit,out=value)
            else:raise ValueError(op)
        values[ref]=value
        for j,slot in enumerate(dest):
            if op=='input'or j:fnp.copyto(outputs[slot],value)
        for a in args:
            remaining[a]-=1
            if remaining[a]==0:
                if a in owners:owners.remove(a);stats['temp_live']-=math.prod(shape)*4
                values.pop(a)
        if remaining[ref]==0:
            if ref in owners:owners.remove(ref);stats['temp_live']-=math.prod(shape)*4
            values.pop(ref)
    assert not owners and not values and stats['temp_live']==0

class Kernel:
    def __init__(self,module,record):
        self.M=module;self.fnp=module.fnp;self.record=record
        self.circuits={k:prepare(record['circuits'][k])for k in ('L','R','P')}
        self.cache={}
        self.stats={}
    def side(self,kind,L,P,rows,cols):
        key=(kind,L,P,rows,cols)
        if key in self.cache:return self.cache[key]
        fnp=self.fnp;M=self.M;u=1<<L;r,c=rows//u,cols//u
        T=fnp.empty((7,)*L+(P,r,c),dtype=fnp.float32)
        self.stats['leaf_bytes']+=math.prod(T.shape)*4
        data=slice(0,7,2)if kind==0 else slice(0,4)
        chunks=[None,None]
        for j in range(2,L+1):
            nr=L-j;chunks.append(fnp.unstack(T[(data,)*nr]if nr else T,axis=nr))
        nr=L-1;R=T[(data,)*nr]
        V1=fnp.reshape(R,(2,2)*nr+(7,P,r,c))
        swap=[x for q in range(nr)for x in (2*q+1,2*q)]if kind==2 else list(range(2*nr))
        base=swap+list(range(2*nr,2*nr+4))
        G=fnp.transpose(V1,[base[x]for x in M._rs_nat_perm(L,1)])
        chunks[1]=fnp.unstack(G,axis=0)
        value=(T,chunks,G,M._rs_basis_views(kind,L,T))
        self.cache[key]=value;return value
    def bb_batch(self,A,B,C,L):
        fnp=self.fnp;P,m,k=map(int,A.shape);n=int(B.shape[-1]);u=1<<L
        TA,XA,GA,basA=self.side(0,L,P,m,k)
        TB,XB,GB,basB=self.side(1,L,P,k,n)
        TC,XC,GC,basC=self.side(2,L,P,m,n)
        Ar=fnp.reshape(A,(P,)+(2,)*L+(m//u,)+(2,)*L+(k//u,))
        Br=fnp.reshape(B,(P,)+(2,)*L+(k//u,)+(2,)*L+(n//u,))
        corner=lambda V,i,j:V[(slice(None),i)+(slice(None),)*L+(j,)]
        X=XA[1];Y=XB[1]
        fnp.copyto(fnp.moveaxis(GA[0:3:2],0,L+1),Ar[:,0])
        a21=corner(Ar,1,0);a22=corner(Ar,1,1)
        fnp.copyto(X[6],a22);fnp.add(a21,a22,out=X[4])
        left=(slice(None),)*(L+2)+(0,)
        fnp.copyto(fnp.moveaxis(GB[0:3:2],0,1),Br[left])
        b12=corner(Br,0,1);b22=corner(Br,1,1)
        fnp.copyto(Y[3],b22);fnp.subtract(b22,b12,out=Y[1])
        for lo,hi in basA:fnp.add(lo,hi,out=lo)
        for lo,hi in basB:fnp.subtract(hi,lo,out=lo)
        for j in range(1,L+1):
            X=XA[j];Y=XB[j]
            fnp.subtract(X[4],X[0],out=X[5]);fnp.subtract(X[2],X[5],out=X[3]);fnp.subtract(X[6],X[5],out=X[1])
            fnp.add(Y[1],Y[0],out=Y[5]);fnp.subtract(Y[5],Y[2],out=Y[6]);fnp.subtract(Y[3],Y[5],out=Y[4])
        fnp.matmul(TA,TB,out=TC)
        for j in range(L,1,-1):
            Z=XC[j]
            fnp.add(Z[0],Z[5],out=Z[5]);fnp.add(Z[0],Z[2],out=Z[0]);fnp.add(Z[5],Z[4],out=Z[5])
            fnp.add(Z[5],Z[3],out=Z[2]);fnp.add(Z[5],Z[1],out=Z[3]);fnp.add(Z[4],Z[6],out=Z[1])
        for lo,hi in basC:fnp.subtract(hi,lo,out=lo)
        Cr=fnp.reshape(C,(P,)+(2,)*L+(m//u,)+(2,)*L+(n//u,))
        tl,tr,bl,br=(corner(Cr,i,j)for i,j in ((0,0),(0,1),(1,0),(1,1)))
        Z=XC[1]
        fnp.add(Z[0],Z[5],out=Z[5]);fnp.add(Z[0],Z[2],out=tl);fnp.add(Z[5],Z[4],out=Z[5])
        fnp.add(Z[5],Z[3],out=tr);fnp.add(Z[5],Z[1],out=br);fnp.add(Z[4],Z[6],out=Z[4]);fnp.subtract(br,Z[4],out=bl)
        return C
    def __call__(self,a,b,L,P,old=None):
        M=self.M;fnp=self.fnp;m,k=int(a.shape[0]),int(a.shape[1]);n=int(b.shape[1])
        rank=0 if old is None else int(old[0].shape[1]);ar,ak,an=self.record['shape'];u=1<<L
        cm=math.ceil(m/ar/u)*u;ck=math.ceil((k+rank)/ak/u)*u;cn=math.ceil(n/an/u)*u
        assert 40%P==0 and tuple(b.shape)==(k,n)
        sizes=[7**L*x*y//u**2 for x,y in ((cm,ck),(ck,cn),(cm,cn))]
        assert all(P*x<=y for x,y in zip(sizes,M._IPT_REGION[L]))
        self.cache.clear();self.stats=dict(temp_live=0,temp_peak=0,leaf_bytes=0)
        aa=fnp.zeros((ar*cm,ak*ck),dtype=fnp.float32);bb=fnp.zeros((ak*ck,an*cn),dtype=fnp.float32)
        fnp.copyto(aa[:m,:k],a);fnp.copyto(bb[:k,:n],b)
        if old is not None:
            fnp.copyto(aa[:m,k:k+rank],old[0]);fnp.copyto(bb[k:k+rank,:n],old[1])
        LA=fnp.empty((40,cm,ck),dtype=fnp.float32);RB=fnp.empty((40,ck,cn),dtype=fnp.float32);PC=fnp.empty((40,cm,cn),dtype=fnp.float32)
        la=list(fnp.unstack(LA,axis=0));rb=list(fnp.unstack(RB,axis=0));pc=list(fnp.unstack(PC,axis=0))
        av=[aa[i*cm:(i+1)*cm,j*ck:(j+1)*ck]for i in range(ar)for j in range(ak)]
        bv=[bb[i*ck:(i+1)*ck,j*cn:(j+1)*cn]for i in range(ak)for j in range(an)]
        linear(fnp,self.circuits['L'],av,la,(cm,ck),self.stats)
        linear(fnp,self.circuits['R'],bv,rb,(ck,cn),self.stats)
        self.stats['packed_bytes']=(40*(cm*ck+ck*cn+cm*cn)+(ar*cm)*(ak*ck)+(ak*ck)*(an*cn))*4
        try:
            for start in range(0,40,P):self.bb_batch(LA[start:start+P],RB[start:start+P],PC[start:start+P],L)
        finally:
            self.cache.clear();M._BH_HOLD.clear()
        out=fnp.empty((ar*cm,an*cn),dtype=fnp.float32)
        ov=[out[i*cm:(i+1)*cm,j*cn:(j+1)*cn]for i in range(ar)for j in range(an)]
        linear(fnp,self.circuits['P'],pc,ov,(cm,cn),self.stats)
        self.stats['conservative_peak_bytes']=self.stats['leaf_bytes']+self.stats['packed_bytes']+self.stats['temp_peak']+int(out.shape[0])*int(out.shape[1])*4
        return fnp.copy(out[:m,:n])if tuple(out.shape)!=(m,n)else out
