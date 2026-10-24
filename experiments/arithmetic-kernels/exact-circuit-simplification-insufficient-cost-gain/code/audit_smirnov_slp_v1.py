"""Whitelist parser for public Maple algebra DATA, never eval/exec source.

Counts scalar arithmetic by linear stage, checks every elementary matrix
product, and prices fixed small hybrid schedules. No large matrix run.
"""
import ast
import bz2
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import numpy as np

HERE=Path(__file__).resolve().parent
SRC=HERE/'public_fmm_sources_v1/3x3x6_raw.mpl.bz2'
OUT=HERE/'smirnov_slp_audit_v1.json';assert not OUT.exists()
raw=bz2.decompress(SRC.read_bytes()).decode('ascii')
body=raw.split('C := Matrix(3,6);',1)[1].split('return C end proc;',1)[0]
program=[]
for item in body.split(';'):
    if not item.strip():continue
    name,expression=item.strip().split(' := ')
    expression=expression.replace('LinearAlgebra:-MatrixMatrixMultiply','mm')
    program.append((name,ast.parse(expression,mode='eval').body))


def run(A,B):
    values={};counts=dict(A=0,B=0,C=0);products=0
    def expr(node):
        nonlocal products
        if isinstance(node,ast.Constant):
            assert type(node.value) is int
            return Fraction(node.value),None
        if isinstance(node,ast.Name):return values[node.id]
        if isinstance(node,ast.Subscript):
            assert isinstance(node.value,ast.Name) and node.value.id in ('A','B')
            assert isinstance(node.slice,ast.Tuple)
            ids=[x.value-1 for x in node.slice.elts]
            arr=A if node.value.id=='A' else B
            return arr[tuple(ids)],node.value.id
        if isinstance(node,ast.UnaryOp):
            v,stage=expr(node.operand);assert isinstance(node.op,ast.USub)
            if stage:counts[stage]+=1
            return -v,stage
        if isinstance(node,ast.BinOp):
            a,sa=expr(node.left);b,sb=expr(node.right)
            stage=sa or sb
            assert sa is None or sb is None or sa==sb
            if stage:counts[stage]+=1
            if isinstance(node.op,ast.Add):v=a+b
            elif isinstance(node.op,ast.Sub):v=a-b
            elif isinstance(node.op,ast.Mult):
                assert sa is None or sb is None
                v=a*b
            elif isinstance(node.op,ast.Div):
                assert sb is None
                v=a/b
            else:raise TypeError(type(node.op))
            return v,stage
        if isinstance(node,ast.Call):
            assert isinstance(node.func,ast.Name) and node.func.id=='mm' and len(node.args)==2
            a,sa=expr(node.args[0]);b,sb=expr(node.args[1]);assert (sa,sb)==('A','B')
            products+=1;return a*b,'C'
        raise TypeError(type(node))
    C=np.zeros((3,6),dtype=object)
    for name,node in program:
        v,stage=expr(node)
        match=re.fullmatch(r'C\[(\d),(\d)\]',name)
        if match:
            i,j=map(int,match.groups());C[i-1,j-1]=v
        values[name]=(v,stage)
        # Output elements may be referred to later by Maple subscripting.
        if match:values[f'output_{i}_{j}']=(v,stage)
    return C,counts,products

# Replace only output-element references with whitelisted scalar names.
class OutputNames(ast.NodeTransformer):
    def visit_Subscript(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='C':
            i,j=[x.value for x in node.slice.elts]
            return ast.copy_location(ast.Name(id=f'output_{i}_{j}',ctx=ast.Load()),node)
        return node
program=[(name,OutputNames().visit(node)) for name,node in program]
maxerror=0
for ia,ib in itertools.product(range(9),range(18)):
    A=np.zeros((3,3),dtype=object);B=np.zeros((3,6),dtype=object)
    A.flat[ia]=Fraction(1);B.flat[ib]=Fraction(1)
    C,counts,products=run(A,B)
    maxerror=max(maxerror,max(abs(x) for x in (C-A@B).flat))
assert maxerror==0 and products==40

coef=json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
def leaf(m,k,n,d):
    if d==0:return 2*m*k*n-m*n
    u=2**d;m,k,n=((x+u-1)//u*u for x in (m,k,n))
    return sum(a*b for a,b in zip(coef[str(d)],(m*k*n,m*k,k*n,m*n)))


@lru_cache(None)
def price(m,k,n,stages,d):
    if not stages:return leaf(m,k,n,d)
    mm,kk,nn=(m+2)//3,(k+2)//3,(n+5)//6
    return 40*price(mm,kk,nn,stages-1,d)+counts['A']*mm*kk+counts['B']*kk*nn+counts['C']*mm*nn

shapes=((1024,864,8192),(1024,832,8192),(864,8192,864),(832,8192,832),(1024,1024,1024))
rows=[]
for m,k,n in shapes:
    baseline=leaf(m,k,n,4) if max(m,k,n)>1024 else min(leaf(m,k,n,d) for d in (3,4,5))
    options=[]
    # Only actual primary orientation and its exact matrix-transpose dual;
    # do not invent cyclic-dual addition counts.
    for stages in (1,2):
        for d in (0,3,4,5):
            options.append(dict(stages=stages,winograd_depth=d,flops=price(m,k,n,stages,d),orientation='3x3x6'))
            # Cost of B.T@A.T, swapping input-stage addition weights.
            @lru_cache(None)
            def transpose_price(m,k,n,s):
                if not s:return leaf(m,k,n,d)
                mm,kk,nn=(m+5)//6,(k+2)//3,(n+2)//3
                return 40*transpose_price(mm,kk,nn,s-1)+counts['B']*mm*kk+counts['A']*kk*nn+counts['C']*mm*nn
            options.append(dict(stages=stages,winograd_depth=d,flops=transpose_price(m,k,n,stages),orientation='6x3x3'))
    best=min(options,key=lambda x:x['flops'])
    rows.append(dict(shape=(m,k,n),bank_kernel_model=baseline,best=best,ratio=best['flops']/baseline,
                     options=options))
result=dict(exact_basis_products=162,max_exact_error=str(maxerror),multiplications=products,
            scalar_linear_arithmetic=counts,compressed_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
            source_url='https://fmm.univ-lille.fr/3x3x6_raw.mpl.bz2',rows=rows,
            qualification='Shape-only primary-SLP arithmetic; includes repeated child padding but omits copies, allocations, live-memory and remote overhead. Optimistic, NOT native bill or proposed estimator.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k!='options'} for r in rows]},indent=2))
