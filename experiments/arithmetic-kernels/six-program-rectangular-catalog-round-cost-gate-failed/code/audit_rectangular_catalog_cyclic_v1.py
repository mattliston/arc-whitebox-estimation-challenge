"""Exact bilinear tensors for the missing cyclic catalog orientations."""
import ast
import bz2
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'rectangular_catalog_cyclic_certificate_v1.json'
assert not OUT.exists()
local=(HERE/'audit_rectangular_catalog_v1.py').read_text().split('start=time.time()',1)[0]
local=local.replace('assert not OUT.exists()','')
ns=dict(__file__=str(HERE/'audit_rectangular_catalog_v1.py'))
exec(compile(local,'local_restricted_parser','exec'),ns)
old=json.loads((HERE/'rectangular_catalog_certificate_v1.json').read_text())
assert old['passed']


def tensors(program,shape,rank):
    m,k,n=shape;values={};left=[];right=[]
    A=np.eye(m*k,dtype=object).reshape(m,k,m*k)
    B=np.eye(k*n,dtype=object).reshape(k,n,k*n)
    def multiply(a,b):
        left.append(a);right.append(b)
        v=np.zeros(rank,dtype=object);v[len(left)-1]=Fraction(1)
        return v,'C'
    def expr(node):
        if isinstance(node,ast.Constant):
            assert type(node.value) is int
            return Fraction(node.value),None
        if isinstance(node,ast.Name):return values[node.id]
        if isinstance(node,ast.Subscript):
            indices=tuple(x.value-1 for x in node.slice.elts)
            stage=node.value.id
            if stage=='A':return A[indices],stage
            if stage=='B':return B[indices],stage
            assert stage=='C'
            return values[f'C[{indices[0]+1},{indices[1]+1}]']
        if isinstance(node,ast.UnaryOp):
            assert isinstance(node.op,ast.USub)
            v,s=expr(node.operand);return -v,s
        if isinstance(node,ast.Call):
            assert node.func.id=='mm' and len(node.args)==2
            (a,sa),(b,sb)=map(expr,node.args)
            assert (sa,sb)==('A','B');return multiply(a,b)
        if isinstance(node,ast.BinOp):
            a,sa=expr(node.left);b,sb=expr(node.right)
            if isinstance(node.op,ast.Mult) and (sa,sb)==('A','B'):
                return multiply(a,b)
            assert sa is None or sb is None or sa==sb
            stage=sa or sb
            if isinstance(node.op,ast.Add):return a+b,stage
            if isinstance(node.op,ast.Sub):return a-b,stage
            if isinstance(node.op,ast.Mult):
                assert sa is None or sb is None
                return a*b,stage
            if isinstance(node.op,ast.Div):
                assert sb is None
                return a/b,stage
        raise TypeError(type(node))
    for name,node in program:values[name]=expr(node)
    assert len(left)==rank
    P=np.stack([values[f'C[{i},{j}]'][0] for i in range(1,m+1) for j in range(1,n+1)])
    return np.stack(left),np.stack(right),P


def ct(A,m,n):
    return A.reshape(A.shape[0],m,n).transpose(0,2,1).reshape(A.shape[0],m*n)


def rt(A,m,n):
    return A.reshape(m,n,A.shape[1]).transpose(1,0,2).reshape(m*n,A.shape[1])


start=time.time();records=[]
for rec in old['records']:
    shape=tuple(rec['shape']);m,k,n=shape;rank=rec['rank']
    name='x'.join(map(str,shape))
    data=(HERE/'public_rectangular_catalog_data_v1'/(name+'_raw.mpl.bz2')).read_bytes()
    assert hashlib.sha256(data).hexdigest()==rec['compressed_sha256']
    program,kind=ns['parse'](bz2.decompress(data).decode('ascii'),shape)
    L,R,P=tensors(program,shape,rank)
    a,b,c=rec['arithmetic']
    entries=[dict(name=name+'_primary',shape=shape,L=L,R=R,P=P,arithmetic=(a,b,c)),
             dict(name=name+'_cyclic1',shape=(k,n,m),L=R,R=ct(P.T,m,n),P=rt(L.T,m,k),
                  arithmetic=(b,c+m*n-rank,a+rank-m*k)),
             dict(name=name+'_cyclic2',shape=(n,m,k),L=ct(P.T,m,n),R=L,P=rt(R.T,k,n),
                  arithmetic=(c+m*n-rank,a,b+rank-k*n))]
    for e in entries.copy():
        em,ek,en=e['shape'];ea,eb,ec=e['arithmetic']
        entries.append(dict(name=e['name']+'_transpose',shape=(en,ek,em),
                            L=ct(e['R'],ek,en),R=ct(e['L'],em,ek),P=rt(e['P'],em,en),
                            arithmetic=(eb,ea,ec)))
    for e in entries:
        em,ek,en=e['shape'];l,r,p=e['L'],e['R'],e['P']
        value=np.zeros((em*ek,ek*en,em*en),dtype=object)
        for j in range(rank):value+=l[j,:,None,None]*r[j,None,:,None]*p[:,j][None,None,:]
        target=np.zeros_like(value)
        for i in range(em):
            for q in range(ek):
                for j in range(en):target[i*ek+q,q*en+j,i*en+j]=1
        assert np.array_equal(value,target),e['name']
        records.append(dict(name='catalog_'+e['name'],shape=e['shape'],rank=rank,
                            arithmetic=e['arithmetic'],exact_basis_products=em*ek*ek*en,
                            passed=True))
    print(json.dumps(dict(source=name,certified_orientations=6,wall_s=time.time()-start)),flush=True)
result=dict(passed=True,records=records,wall_s=time.time()-start,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Exact rational bilinear tensors; transposition arithmetic envelopes, not implemented circuits/copy counts/memory/runtime. Same six primary data files, no new download or estimator.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
