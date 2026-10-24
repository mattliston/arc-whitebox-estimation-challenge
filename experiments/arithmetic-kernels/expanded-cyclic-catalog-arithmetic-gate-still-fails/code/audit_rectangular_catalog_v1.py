"""Restricted scalar arithmetic parser; exact basis certificate, no eval."""
import ast
import bz2
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import time
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'rectangular_catalog_certificate_v1.json'
assert not OUT.exists()
CASES = (((2,3,3),15),((2,4,5),32),((2,4,8),51),
         ((3,4,4),38),((3,4,5),47),((3,5,5),58))


def parse(raw, shape):
    m,k,n = shape
    if 'MUL:=[' in raw:
        mul = raw.split('MUL:=[',1)[1].split(']:',1)[0]
        add = raw.split('ADD:=[',1)[1].split(']:',1)[0]
        statements = []
        for line in (mul+','+add).split(','):
            if not line.strip():
                continue
            left, right = line.strip().split('=')
            if left == '0':
                assert right == '0'
                continue
            left = re.sub(r'C_(\d+)_(\d+)',r'C[\1,\2]',left)
            right = re.sub(r'([AB])_(\d+)_(\d+)',r'\1[\2,\3]',right)
            statements.append((left, ast.parse(right,mode='eval').body))
        return statements, 'literal_MUL_ADD'
    header = re.search(r'C\s*:=\s*Matrix\(\s*'+str(m)+r'\s*,\s*'+str(n)+r'\s*\);',raw)
    assert header is not None
    body = raw[header.end():].split('return C end proc;',1)[0]
    statements = []
    for item in body.split(';'):
        if not item.strip():
            continue
        left, right = item.strip().split(' := ')
        right = right.replace('LinearAlgebra:-MatrixMatrixMultiply','mm')
        statements.append((left,ast.parse(right,mode='eval').body))
    return statements, 'published_shared_SLP'


def run(program, A, B):
    values = {}
    counts = dict(A=0,B=0,C=0)
    mm_count = 0
    def expr(node):
        nonlocal mm_count
        if isinstance(node,ast.Constant):
            assert type(node.value) is int
            return Fraction(node.value), None
        if isinstance(node,ast.Name):
            return values[node.id]
        if isinstance(node,ast.Subscript):
            assert isinstance(node.value,ast.Name) and node.value.id in ('A','B','C')
            assert isinstance(node.slice,ast.Tuple) and len(node.slice.elts)==2
            assert all(isinstance(x,ast.Constant) and type(x.value) is int for x in node.slice.elts)
            i,j = (x.value-1 for x in node.slice.elts)
            if node.value.id == 'C':
                return values[f'C[{i+1},{j+1}]']
            arr = A if node.value.id=='A' else B
            return arr[i,j], node.value.id
        if isinstance(node,ast.UnaryOp):
            assert isinstance(node.op,ast.USub)
            value, stage = expr(node.operand)
            if stage:
                counts[stage] += 1
            return -value, stage
        if isinstance(node,ast.Call):
            assert isinstance(node.func,ast.Name) and node.func.id=='mm' and len(node.args)==2 and not node.keywords
            a,sa = expr(node.args[0]); b,sb = expr(node.args[1])
            assert (sa,sb)==('A','B')
            mm_count += 1
            return a*b,'C'
        if isinstance(node,ast.BinOp):
            a,sa = expr(node.left); b,sb = expr(node.right)
            if isinstance(node.op,ast.Mult) and (sa,sb)==('A','B'):
                mm_count += 1
                return a*b,'C'
            stage = sa or sb
            assert sa is None or sb is None or sa==sb
            if stage:
                counts[stage] += 1
            if isinstance(node.op,ast.Add): value=a+b
            elif isinstance(node.op,ast.Sub): value=a-b
            elif isinstance(node.op,ast.Mult):
                assert sa is None or sb is None
                value=a*b
            elif isinstance(node.op,ast.Div):
                assert sb is None
                value=a/b
            else: raise TypeError(type(node.op))
            return value,stage
        raise TypeError(type(node))
    C = np.zeros((A.shape[0],B.shape[1]),dtype=object)
    for name,node in program:
        value,stage = expr(node)
        values[name] = value,stage
        match = re.fullmatch(r'C\[(\d+),(\d+)\]',name)
        if match:
            i,j = map(int,match.groups())
            assert stage=='C'
            C[i-1,j-1] = value
    return C,counts,mm_count


start=time.time()
records=[]
for shape,rank in CASES:
    name='x'.join(map(str,shape))
    path=HERE/'public_rectangular_catalog_data_v1'/(name+'_raw.mpl.bz2')
    compressed=path.read_bytes()
    program,kind=parse(bz2.decompress(compressed).decode('ascii'),shape)
    m,k,n=shape
    for ia,ib in product(range(m*k),range(k*n)):
        A=np.zeros((m,k),dtype=object);B=np.zeros((k,n),dtype=object)
        A.flat[ia]=Fraction(1);B.flat[ib]=Fraction(1)
        C,counts,products=run(program,A,B)
        assert products==rank and np.array_equal(C,A@B),(shape,ia,ib,products)
    rec=dict(shape=shape,rank=rank,kind=kind,arithmetic=[counts[s] for s in ('A','B','C')],
             exact_basis_products=m*k*k*n,passed=True,compressed_sha256=hashlib.sha256(compressed).hexdigest())
    records.append(rec)
    print(json.dumps(rec),flush=True)
result=dict(passed=True,records=records,wall_s=time.time()-start,
            qualification='Exact rational basis products; arithmetic of the actual published shared SLP or literal MUL/ADD expression list, as labelled. No downloaded execution, copies, memory schedule, floating-point stability or runtime claim.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
