"""Exact tensor certificates for all cyclic/transpose duals of the public SLP.

Downloaded Maple is parsed as whitelisted algebra data, never executed.
The reused parser/certificate below is our own already-authored local code.
"""
import ast
from fractions import Fraction
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'smirnov_cyclic_certificate_v1.json';assert not OUT.exists()
own_source=(HERE/'audit_smirnov_slp_v1.py').read_text().split('coef=json.loads',1)[0]
needle="OUT=HERE/'smirnov_slp_audit_v1.json';assert not OUT.exists()"
assert own_source.count(needle)==1
own_source=own_source.replace(needle,'OUT=None')
namespace=dict(__file__=str(HERE/'audit_smirnov_slp_v1.py'),__name__='reused_whitelist_algebra')
exec(compile(own_source,str(HERE/'audit_smirnov_slp_v1.py'),'exec'),namespace)
program=namespace['program'];counts=namespace['counts'];values={};left=[];right=[]


def expr(node):
    if isinstance(node,ast.Constant):return Fraction(node.value)
    if isinstance(node,ast.Name):return values[node.id]
    if isinstance(node,ast.Subscript):
        ids=tuple(x.value-1 for x in node.slice.elts)
        if node.value.id=='A':return np.eye(9,dtype=object).reshape(3,3,9)[ids]
        if node.value.id=='B':return np.eye(18,dtype=object).reshape(3,6,18)[ids]
        raise TypeError(node.value.id)
    if isinstance(node,ast.UnaryOp):
        assert isinstance(node.op,ast.USub);return -expr(node.operand)
    if isinstance(node,ast.BinOp):
        a,b=expr(node.left),expr(node.right)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
        raise TypeError(type(node.op))
    if isinstance(node,ast.Call):
        assert node.func.id=='mm'
        left.append(expr(node.args[0]));right.append(expr(node.args[1]))
        v=np.zeros(40,dtype=object);v[len(left)-1]=Fraction(1);return v
    raise TypeError(type(node))


for name,node in program:
    value=expr(node);values[name]=value
    if name.startswith('C['):
        ij=name[2:-1].split(',');values[f'output_{ij[0]}_{ij[1]}']=value
L=np.stack(left);R=np.stack(right)
P=np.stack([values[f'C[{i},{j}]'] for i in range(1,4) for j in range(1,7)])
assert L.shape==(40,9) and R.shape==(40,18) and P.shape==(18,40)


def col_transpose(A,m,n):return A.reshape(A.shape[0],m,n).transpose(0,2,1).reshape(A.shape[0],m*n)
def row_transpose(A,m,n):return A.reshape(m,n,A.shape[1]).transpose(1,0,2).reshape(m*n,A.shape[1])


# Linear transposition: additions(T^T)=additions(T)+outputs-inputs;
# scalar multiplications/divisions are retained. This is the arithmetic
# envelope, not a claim about copies, materialization, or live-memory.
ca,cb,cc=counts['A'],counts['B'],counts['C']
programs=[dict(name='primary_3x3x6',shape=(3,3,6),L=L,R=R,P=P,arithmetic=(ca,cb,cc)),
          dict(name='cyclic_3x6x3',shape=(3,6,3),L=R,R=col_transpose(P.T,3,6),P=row_transpose(L.T,3,3),
               arithmetic=(cb,cc+18-40,ca+40-9)),
          dict(name='cyclic_6x3x3',shape=(6,3,3),L=col_transpose(P.T,3,6),R=L,P=row_transpose(R.T,3,6),
               arithmetic=(cc+18-40,ca,cb+40-18))]
for entry in programs.copy():
    m,k,n=entry['shape'];a,b,c=entry['arithmetic']
    programs.append(dict(name=entry['name']+'_transpose',shape=(n,k,m),
                         L=col_transpose(entry['R'],k,n),R=col_transpose(entry['L'],m,k),
                         P=row_transpose(entry['P'],m,n),arithmetic=(b,a,c)))
receipts=[]
for entry in programs:
    m,k,n=entry['shape'];l,r,p=entry['L'],entry['R'],entry['P']
    tensor=np.zeros((m*k,k*n,m*n),dtype=object)
    for j in range(40):tensor+=l[j,:,None,None]*r[j,None,:,None]*p[:,j][None,None,:]
    reference=np.zeros_like(tensor)
    for i in range(m):
        for q in range(k):
            for j in range(n):reference[i*k+q,q*n+j,i*n+j]=1
    error=max(abs(x) for x in (tensor-reference).flat)
    assert error==0
    receipts.append(dict(name=entry['name'],shape=entry['shape'],rank=40,
                         elementary_input_pairs=m*k*k*n,max_exact_error=str(error),
                         scalar_linear_arithmetic=entry['arithmetic'],
                         L=[[str(x) for x in row] for row in l],R=[[str(x) for x in row] for row in r],
                         P=[[str(x) for x in row] for row in p]))
result=dict(passed=True,programs=receipts,source='https://fmm.univ-lille.fr/3x3x6_raw.mpl.bz2',
            qualification='Exact rational bilinear tensors for all six cyclic/transpose orientations. Linear-transposition arithmetic envelope excludes copies/allocations and is not a native measured implementation.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'programs':[{k:v for k,v in x.items() if k not in ('L','R','P')} for x in receipts]},indent=2))
