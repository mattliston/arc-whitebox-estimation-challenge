"""Fixed one-stage certified Smirnov + current batched binary kernel.

No numerical library is imported here. The native or symbolic backend is injected.
"""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from functools import lru_cache
from math import prod
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
ASTRA = H2.parent / 'astra_phase2_reexamination_v1'
PARENT = H2 / 'candidate_old_both_tail_trace_bounded_v6.py'
CERT = ASTRA / 'smirnov_executable_transpose_certificate_v1.json'
OLD = ASTRA / 'smirnov_stream8_native_v1.py'
PARENT_SHA = '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
CERT_SHA = '005e9ddbecae84a4249d1a977ec692ca6a488af182841b3c162aa368d96231b6'
OLD_SHA = '7c9ca40924410a7d5b6280c40bf500f78d9c935c831914e90047c549de430c74'

CASES = (
    dict(name='read6016', shape=(672,6016,864), child=(224,1024,288),
         circuit='signed_executable_1_transpose', census=(35,237)),
    dict(name='read8192', shape=(672,8192,864), child=(224,1376,288),
         circuit='signed_executable_1_transpose', census=(100,594)),
    dict(name='common6144', shape=(672,6144,384), child=(224,1024,128),
         circuit='signed_executable_1', census=(35,184)),
)


def checked(path, sha):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, path
    return data.decode()


@lru_cache(None)
def functions(source):
    result = {}
    lines=source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            first=min([node.lineno]+[d.lineno for d in node.decorator_list])
            result.setdefault(node.name, []).append(''.join(lines[first-1:node.end_lineno]))
    return result


def region(shape, batch=1):
    m,k,n = shape
    assert all(x % 32 == 0 for x in shape)
    return tuple(7**5 * batch * a//32 * (b//32) for a,b in ((m,k),(k,n),(m,n)))


def binary_environment(fnp, shared, shape, batch):
    source = checked(PARENT, PARENT_SHA)
    defs = functions(source)
    env = dict(fnp=fnp, _SCRATCH=(shared,), _IPT_REGION={5:region(shape,batch)},
               _BH_HOLD=[], _CAP_BY_ID={}, _AB_ON=True,
               _LVF_EAGER=[True], _LVF_KEEP=[False], _LVF_PENDING={},
               _LVF_STATS={'fused_keys':0})
    for name in ('_IPT_A','_IPT_B','_IPT_C','_IPT_BA','_IPT_BB','_IPT_BC'):
        env[name] = {}
    # All kernel bodies are extracted verbatim. Baseline's P6 planner rejects
    # these three census shapes (m,n <=1024 and k>=2048); use its exact core.
    for name in ('_cache_put','_ipt_view','_ipt_bits','_rs_nat_perm',
                 '_rs_basis_views','_rs_level1','_ipt_side_p',
                 '_rs_load_a','_rs_load_b','_rs_out_quadrants',
                 '_lvf_rs_level1','_lvf_at','_lvf_front','_lvf_ops','_ipt_side',
                 '_ipt_forward_b_ab','_ipt_leaf_unwind_ab','_bb_product'):
        exec(compile(defs[name][-1], str(PARENT)+':'+name, 'exec'),env)
    # The final forward-A is an identity reuse wrapper. These independent
    # fixture calls require a fresh A transform, as do the selected census rows.
    exec(compile(defs['_ipt_forward_a_ab'][-2], str(PARENT)+':forwardA', 'exec'),env)
    exec(compile(defs['_ipt_product_ab'][-2], str(PARENT)+':product', 'exec'),env)
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and
              v.id in ('_LVF_A','_LVF_B','_LVF_C') for v in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(PARENT),'exec'),env)
    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed current batched kernel unexpectedly fell back')
    env['_AB_OLD_BB'] = forbidden
    return env


def old_environment(fnp):
    from dataclasses import dataclass
    defs = functions(checked(OLD, OLD_SHA))
    env = dict(fnp=fnp,Counter=Counter,Fraction=Fraction,prod=prod,dataclass=dataclass,
               __name__=__name__)
    for name in ('Block','Arena','prepare','linear'):
        exec(compile(defs[name][-1],str(OLD)+':'+name,'exec'),env)
    return env


class Hybrid:
    def __init__(self, fnp, shared, case, temporary_capacity=64_000_000):
        self.fnp, self.case = fnp,case
        self.binary = binary_environment(fnp,shared,case['child'],40)
        old = old_environment(fnp)
        record = next(r for r in json.loads(checked(CERT,CERT_SHA))['records']
                      if r['name']==case['circuit'])
        assert record['passed'] and record['rank']==40 and record['shape']==[3,6,3]
        self.circuits = {k:old['prepare'](record['circuits'][k]) for k in ('L','R','P')}
        self.linear = old['linear']
        arena = old['Arena'].__new__(old['Arena'])
        arena.capacity = temporary_capacity
        arena.array = fnp.empty(temporary_capacity,dtype=fnp.float32)
        arena.free = [(0,temporary_capacity)]
        arena.owned={}; arena.views={}; arena.padding={}
        arena.padding_bytes=0; arena.live=arena.peak_live=0
        arena.high_water=0
        allocate = arena.allocate
        def tracked(shape):
            block=allocate(shape)
            arena.high_water=max(arena.high_water,block.start+block.size)
            return block
        arena.allocate=tracked
        self.arena=arena
        a,b,c=case['child']
        self.left=fnp.empty((40,a,b),dtype=fnp.float32)
        self.right=fnp.empty((40,b,c),dtype=fnp.float32)
        self.products=fnp.empty((40,a,c),dtype=fnp.float32)
        self.left3=[self.left[i:i+1] for i in range(40)]
        self.right3=[self.right[i:i+1] for i in range(40)]
        self.product3=[self.products[i:i+1] for i in range(40)]
        self.left2=[self.left[i] for i in range(40)]
        self.right2=[self.right[i] for i in range(40)]
        self.product2=[self.products[i] for i in range(40)]
        self.padding={}

    def pad(self, value, shape, side):
        if tuple(value.shape)==tuple(shape):
            return value
        if side not in self.padding:
            self.padding[side]=self.fnp.zeros(shape,dtype=self.fnp.float32)
        dest=self.padding[side]
        self.fnp.copyto(dest[:value.shape[0],:value.shape[1]],value)
        return dest

    def __call__(self, A, B, out):
        a,b,c=self.case['child']
        M,K,N=3*a,6*b,3*c
        assert tuple(A.shape)+tuple(B.shape[-1:])==tuple(self.case['shape'])
        assert tuple(out.shape)==(M,N)  # all three fixed cases pad only k
        A=self.pad(A,(M,K),'A');B=self.pad(B,(K,N),'B')
        av=[A[None,i*a:(i+1)*a,j*b:(j+1)*b] for i in range(3) for j in range(6)]
        bv=[B[None,i*b:(i+1)*b,j*c:(j+1)*c] for i in range(6) for j in range(3)]
        cv=[out[None,i*a:(i+1)*a,j*c:(j+1)*c] for i in range(3) for j in range(3)]
        self.linear(self.circuits['L'],av,self.left3,(1,a,b),self.arena)
        self.linear(self.circuits['R'],bv,self.right3,(1,b,c),self.arena)
        self.binary['_bb_product'](self.left2,self.right2,self.product2,5)
        self.linear(self.circuits['P'],self.product3,cv,(1,a,c),self.arena)
        assert not self.arena.owned and self.arena.live==0
        return out

    @property
    def workspace_bytes(self):
        a,b,c=self.case['child']
        return 4*(self.arena.capacity+40*(a*b+b*c+a*c)+
                  sum(prod(x.shape) for x in self.padding.values()))
