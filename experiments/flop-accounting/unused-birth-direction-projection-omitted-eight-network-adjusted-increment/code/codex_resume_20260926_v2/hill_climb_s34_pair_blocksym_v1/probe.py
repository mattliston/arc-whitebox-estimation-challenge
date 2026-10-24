"""Root-only paid tiny source-bound pair-series fixture; no model import.
Usage: native_python -B probe.py BUILD_SHA RESULT_PATH (60s/1GB guard).
"""
from pathlib import Path
import ast, hashlib, json, os, sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def extract(parent, candidate, n):
    old={x.name:x for x in ast.parse(parent).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    new={x.name:x for x in ast.parse(candidate).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    env=dict(fnp=fnp,_E2_PAIR_STATE={},_E2_PAIR_ARENA={},_PSP_INDEX={},_PSP_VIEWS={},
             _SCRATCH=[fnp.empty(72*n*n,dtype=fnp.float32)])
    names=['_E2Hold','_e2_pair_state','_e2_pair_arena']
    names2=['_psp_indices','_psp_views','_psp_series','_psp_pair_series']
    exec(compile(ast.Module(body=[old[k] for k in names]+[new[k] for k in names2],type_ignores=[]),'<source-bound helpers>','exec'),env)
    a=parent.index("    Z0, Z1, Z2, Z3, Z4 = st['Zv']\n")
    b=parent.index('    cov, pk21, pk22 = fnp.unstack',a)
    code="def reference(st, n):\n    _e2_H = _E2Hold()\n    _e2_hold = []\n"+parent[a:b]+"    return fnp.sum(st['Z'], axis=0)\n"
    exec(compile(code,'<unchanged parent series>','exec'),env)
    return env

def main():
    assert len(sys.argv)==3
    result_path=Path(sys.argv[2]).resolve(); assert not result_path.exists()
    build_pin=pin(D/'build.json');assert build_pin['sha256']==sys.argv[1]
    build=json.loads((D/'build.json').read_text())
    for key in ('parent','candidate','builder','fragment','protocol','price','primitive'):
        assert build[key]==pin(build[key]['path'])
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    parent=Path(build['parent']['path']).read_text(); candidate=Path(build['candidate']['path']).read_text()
    n=16; block=2; nb=n//block; rows=[]
    with flops.BudgetContext(flop_budget=100_000_000,quiet=True) as ctx:
        env=extract(parent,candidate,n);st=env['_e2_pair_state'](n)
        for kind in ('ordinary','signed_zero','changed_values'):
            t=np.arange(n*n,dtype=np.float32).reshape(n,n)
            rho=((t*3)%17-8)/np.float32(32)
            k22=((t*5)%19-9)/np.float32(16)
            # Only off-block entries are mirrored. Diagonal blocks deliberately remain asymmetric.
            for i in range(nb):
                for j in range(i+1,nb):
                    rho[j*block:(j+1)*block,i*block:(i+1)*block]=rho[i*block:(i+1)*block,j*block:(j+1)*block].T
                    k22[j*block:(j+1)*block,i*block:(i+1)*block]=k22[i*block:(i+1)*block,j*block:(j+1)*block].T
            f=((np.arange(7*n,dtype=np.float32).reshape(7,n)*3)%13-6)/np.float32(8)
            b=((np.arange(7*n,dtype=np.float32).reshape(7,n)*7)%17-8)/np.float32(8)
            if kind=='signed_zero':
                f*=np.float32(0);rho*=np.float32(0);k22*=np.float32(0)
            if kind=='changed_values':
                f+=np.float32(.125);b-=np.float32(.25);rho=-rho;k22=-k22
            ft=np.stack((f,b,b),axis=1);gt=np.stack((f,f,b),axis=1)
            t3=(np.arange(n,dtype=np.float32)%7-3)/np.float32(16)
            t4=(np.arange(n,dtype=np.float32)%5-2)/np.float32(32)
            t21=(t%23-11)/np.float32(64);t31=(t%11-5)/np.float32(32)
            pb=np.stack((np.ones_like(rho),rho,rho*(rho/np.float32(2)),(rho*(rho/np.float32(2)))*(rho/np.float32(3))))
            fnp.copyto(st['FT'],fnp.asarray(ft));fnp.copyto(st['GT'],fnp.asarray(gt))
            fnp.copyto(st['FS1'],fnp.asarray(ft*t3));fnp.copyto(st['GS1'],fnp.asarray(gt*t3))
            fnp.copyto(st['FS2'],fnp.asarray(ft*t4));fnp.copyto(st['GS2'],fnp.asarray(gt*t4))
            fnp.copyto(st['RR'][:1],fnp.asarray(pb[:1,None]));fnp.copyto(st['P16'],fnp.asarray(pb[1:]))
            for key,value in (('T21',t21),('T22',k22),('T31',t31)):fnp.copyto(st[key],fnp.asarray(value))
            before=int(ctx.summary_dict()['flops_used'])
            expected=np.asarray(env['reference'](st,n)).copy()
            reference_flops=int(ctx.summary_dict()['flops_used'])-before
            before=int(ctx.summary_dict()['flops_used'])
            actual=np.asarray(env['_psp_pair_series'](st,n,env['_E2Hold'](),block)).copy()
            actual_flops=int(ctx.summary_dict()['flops_used'])-before
            same(actual,expected)
            for key,value in (('FT',ft),('GT',gt),('P16',pb[1:]),('T21',t21),('T22',k22),('T31',t31)):
                same(np.asarray(st[key]),value)
            assert np.isfinite(actual).all()
            rows.append(dict(case=kind,all_three_channels_byte_exact=True,inputs_unchanged=True,
                             parent_flops=reference_flops,candidate_flops=actual_flops,
                             full_asymmetric_diagonal_blocks=True))
    result=dict(passed=True,build=build_pin,candidate=build['candidate'],parent=build['parent'],probe=pin(__file__),
                cases=rows,model_executed=False,broad_qualified=False,serving_qualified=False)
    with result_path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
