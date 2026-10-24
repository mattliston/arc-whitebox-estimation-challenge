"""Root-only source-bound tiny GN batching fixture; no estimator import.
NATIVE -B probe.py BUILD_SHA RESULT; guard60s/1GB, BLAS1.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent
PARENT_SHA = '6ac194efda6491b48f86eca074493c503f9312c6fd9e667105f378ee7637d6f9'


def pin(path):
    p=Path(path).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def definitions(text):
    return {x.name:x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}


def extract(parent, source, n):
    old=definitions(parent); new=definitions(source)
    env=dict(fnp=fnp,_E2_PAIR_STATE={},_E2_PAIR_ARENA={},_E1GN_CACHE={},
        _SCRATCH=[fnp.empty(72*n*n,dtype=fnp.float32)],_E1GN_PARENT_RESET=lambda saved:None)
    names=['_E2Hold','_e2_pair_state','_e2_pair_arena','_g5_rowmul',
           '_e1gn_state','_e1gn_contract','_pb_reset']
    nodes=ast.parse(source).body
    bodies=[]
    for name in names:
        bodies.append([x for x in nodes if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==name][-1])
    exec(compile(ast.Module(body=bodies,type_ignores=[]),'<pinned definitions>','exec'),env)
    chosen=[]
    for node in old['_gn_dcov'].body:
        if not isinstance(node,ast.Assign):continue
        target=ast.unparse(node.targets[0])
        if target in ('(A1, A2, A3, A4, A5)','t21a','t31a','X'):chosen.append(node)
    assert len(chosen)==4
    code='def parent_kernel(_g5st, rho, dT21, dT31):\n    _e2_H = _E2Hold()\n'
    code+='\n'.join('    '+line for node in chosen for line in ast.unparse(node).splitlines())
    code+='\n    return X\n'
    exec(compile(code,'<actual parent GN statements>','exec'),env)
    # Source-bound factor writes: retain the actual operator/argument order;
    # substitute only supplied precomputed operands for the surrounding formulas.
    writes=[]
    for node in new['_gn_dcov'].body:
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
        call=node.value
        if not (isinstance(call.func,ast.Name) and call.func.id=='_e2_H' and len(call.args)==1):continue
        inner=call.args[0]
        if not isinstance(inner,ast.Call) or not isinstance(inner.func,ast.Attribute):continue
        if inner.func.attr=='divide' and ast.unparse(inner.keywords[0].value)=="_e1gn['D0']":
            inner.args=[ast.Name(id='dK21',ctx=ast.Load()),ast.Name(id='den',ctx=ast.Load())];writes.append(node)
        if inner.func.attr=='multiply' and ast.unparse(inner.keywords[0].value)=="_e1gn['D1T']":
            inner.args[0]=ast.Name(id='weight',ctx=ast.Load());writes.append(node)
    assert len(writes)==2
    code='def child_factors(_e1gn, dK21, den, weight):\n    _e2_H = _E2Hold()\n'
    code+='\n'.join('    '+line for node in writes for line in ast.unparse(node).splitlines())+'\n'
    exec(compile(code,'<actual child GN factor writes>','exec'),env)
    return env


def counts(ctx):
    s=ctx.summary_dict()
    return int(s['flops_used']),{k:int(v.get('calls',v.get('count',0))) for k,v in s['operations'].items()}


def delta(a,b):
    return b[0]-a[0],{k:b[1].get(k,0)-a[1].get(k,0) for k in set(a[1])|set(b[1]) if b[1].get(k,0)!=a[1].get(k,0)}


def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def measure(ctx, fn, array_type, *args):
    views=0; previous=sys.getprofile(); assert previous is None
    def profile(frame,event,arg):
        nonlocal views
        if event=='call' and frame.f_code is array_type.__getitem__.__code__ and frame.f_back.f_code is fn.__code__:
            views+=1
    before=counts(ctx)
    try:
        sys.setprofile(profile); value=fn(*args)
    finally:sys.setprofile(previous)
    return value,delta(before,counts(ctx)),views


def main():
    assert len(sys.argv)==3
    target=Path(sys.argv[2]).resolve();assert not target.exists()
    assert Path(sys.executable).absolute()==D.parent.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    assert np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    bp=pin(D/'build.json');assert bp['sha256']==sys.argv[1]
    b=json.loads((D/'build.json').read_text())
    for key in ('parent','parent_build','comparator','candidate','builder','fragment','protocol','primitive','policy'):
        assert pin(b[key]['path'])==b[key]
    assert b['parent']['sha256']==PARENT_SHA and b['source_inverse_exact'] is True
    parent=Path(b['parent']['path']).read_text();source=Path(b['candidate']['path']).read_text()
    suffix='\n'+Path(b['fragment']['path']).read_text();assert source.endswith(suffix)
    inverse=source[:-len(suffix)]
    for e in reversed(b['edits']):assert inverse.count(e['after'])==1;inverse=inverse.replace(e['after'],e['before'],1)
    assert inverse==parent
    n=8;rng=np.random.default_rng(20260928);cases=[]
    for name in ('normal','random','signed','zero'):
        F=rng.uniform(-2,2,(7,3,n)).astype(np.float32)
        rho=rng.uniform(-.8,.8,(n,n)).astype(np.float32)
        dk=rng.uniform(-2,2,(n,n)).astype(np.float32)
        den=rng.uniform(.5,2,(n,n)).astype(np.float32)
        weight=rng.uniform(-2,2,n).astype(np.float32)
        if name=='normal':F=np.linspace(.125,2,F.size,dtype=np.float32).reshape(F.shape)
        if name=='signed':F=(np.arange(F.size).reshape(F.shape)%9-4).astype(np.float32)/8;rho=np.where(rho>0,np.float32(.5),np.float32(-.25))
        if name=='zero':
            F.fill(0);F.flat[::2]=np.float32(-0.0);rho.fill(0);rho.flat[::2]=np.float32(-0.0)
            dk.fill(0);dk.flat[::2]=np.float32(-0.0);weight.fill(np.float32(-0.0))
        with flops.BudgetContext(flop_budget=10000000,quiet=True) as ctx:
            env=extract(parent,source,n);st=env['_e2_pair_state'](n);env['_e2_pair_arena'](n)
            g=env['_e1gn_state'](st,n);arena=env['_SCRATCH'][0]
            assert env['_e1gn_state'](st,n) is g
            for k in ('D','B','H','D0','D1','D1T','B0'):assert np.shares_memory(np.asarray(g[k]),np.asarray(arena))
            for k in ('left0','left1','right0','right1'):assert np.shares_memory(np.asarray(g[k]),np.asarray(st['FT']))
            assert np.shares_memory(np.asarray(g['rhoT']),np.asarray(st['P16']))
            assert not np.shares_memory(np.asarray(g['D']),np.asarray(g['B']))
            assert not np.shares_memory(np.asarray(g['D']),np.asarray(g['H']))
            assert not np.shares_memory(np.asarray(g['B']),np.asarray(g['H']))
            passes=[]
            for turn in range(2):
                currentF=F if turn==0 else -F;currentR=rho if turn==0 else rho.T.copy()
                fnp.copyto(st['FT'],fnp.asarray(currentF));fnp.copyto(st['P16'][0],fnp.asarray(currentR))
                fnp.copyto(arena,fnp.float32(7.25));neighbor=np.asarray(arena)[6*n*n:].copy()
                fr=fnp.asarray(currentR);fd=fnp.asarray(dk);fden=fnp.asarray(den);fw=fnp.asarray(weight)
                d21=fd/fden;d31=env['_g5_rowmul'](fw,fr)
                expected,old,ov=measure(ctx,env['parent_kernel'],type(st['FT']),st,fr,d21,d31)
                expected=np.asarray(expected).copy()
                env['child_factors'](g,fd,fden,fw);same(g['D0'],d21);same(g['D1'],d31)
                actual,new,nv=measure(ctx,env['_e1gn_contract'],type(st['FT']),g,fr,env['_E2Hold']())
                same(actual,expected);assert np.isfinite(np.asarray(actual)).all()
                same(np.asarray(arena)[6*n*n:],neighbor)
                same(st['FT'],currentF);same(st['P16'][0],currentR)
                arithmetic={'outer','multiply','add','divide','true_divide'}
                assert sum(v for k,v in old[1].items() if k in arithmetic)==13,old
                assert sum(v for k,v in new[1].items() if k in arithmetic)==7,new
                assert old[0]==new[0]==13*n*n,(old,new)
                assert old[1].get('unstack',0)==1 and new[1].get('unstack',0)==0 and ov==1 and nv==0
                passes.append(dict(parent_flops=old[0],child_flops=new[0],parent_operations=old[1],child_operations=new[1],
                    parent_basic_views=ov,child_basic_views=nv,explicit_calls_saved=8,
                    output_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),all_bytes_exact=True))
            # Cached FT views must rebind if their actual owner changes.
            st2=dict(st);st2['FT']=fnp.empty(st['FT'].shape,dtype=fnp.float32)
            g2=env['_e1gn_state'](st2,n);assert g2 is not g
            assert np.shares_memory(np.asarray(g2['left0']),np.asarray(st2['FT']))
            env['_pb_reset']((None,None));assert env['_E1GN_CACHE']=={}
            cases.append(dict(case=name,passes=passes,cached_views_exact=True,neighbor_preserved=True,reset_cleared=True))
    result=dict(passed=True,candidate=b['candidate'],parent=b['parent'],build=bp,probe=pin(__file__),cases=cases,
        arithmetic_calls_before=13,arithmetic_calls_after=7,views_removed=2,per_network_explicit_calls_saved=104,
        source_inverse_exact=True,model_executed=False,release_admitted=False)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
