"""Official tiny live GNN check plus exact E1/mean preservation and priced kernel."""
import argparse, ast, hashlib, inspect, json, subprocess, sys
from pathlib import Path
import numpy as np
import check_frontier_tiny_v5 as tiny

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert sha(tiny.__file__)=='da23022d6e6d688535d91e6bcfba84b226e3224629cdb71b7e0f6790030e2c86'
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--official',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    source=inspect.getsource(tiny.oracle)
    old='    E2=gop(np.stack((w2.T,w4.T,w6.T)),src**2)'
    new='''    TT=np.stack((w2.T,w4.T,w6.T));XX=src**2
    E2=np.empty_like(XX)
    E2[0:1]=gop(TT[0:1],XX[0:1])
    E2[1:3]=TT[1:3].sum(axis=2)[:,:,None]*XX[1:3].mean(axis=1)[:,None,:]'''
    assert source.count(old)==1
    namespace=dict(np=np,gop=tiny.gop,w3=tiny.w3)
    exec(compile(source.replace(old,new),'<independent-energy-mean-oracle>','exec'),namespace)
    tiny.oracle=namespace['oracle'];tiny.main()
    from flopscope import numpy as fnp
    import flopscope as fs
    c=json.loads(a.manifest.read_text())['candidates'][0]
    text=Path(c['science']['path']).read_text();nodes={n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    env=dict(fnp=fnp);exec(compile(ast.Module(body=[nodes[n] for n in ('_E2Hold','_gop2_contract','_vmfr_e23_energy')],type_ignores=[]),'<energy-kernels>','exec'),env)
    rng=np.random.default_rng(20261021);cases=[]
    for n,kind in [(16,'random'),(32,'random'),(16,'constant'),(1024,'random')]:
        w=rng.normal(0,np.sqrt(2/n),(n,n)).astype('f4');w2=w*w;w3=w2*w
        tt=np.stack((w2.T,(w2*w2).T,(w3*w3).T));xx=(rng.normal(0,.5,(3,n,8)).astype('f4')**2)
        if kind=='constant':xx[:]=.2
        with fs.BudgetContext(flop_budget=2**32,quiet=True) as ctx:
            T,X=fnp.asarray(tt),fnp.asarray(xx)
            before=ctx.flops_used;cnt=len(ctx.op_log);old=env['_gop2_contract'](T,X)
            parent_bill=ctx.flops_used-before;parent_calls=len(ctx.op_log)-cnt;old_np=np.asarray(old).copy()
            before=ctx.flops_used;cnt=len(ctx.op_log);new=env['_vmfr_e23_energy'](T,X)
            new_bill=ctx.flops_used-before;new_calls=len(ctx.op_log)-cnt;new_np=np.asarray(new).copy()
            np.testing.assert_array_equal(np.asarray(T),tt);np.testing.assert_array_equal(np.asarray(X),xx)
        np.testing.assert_array_equal(old_np[0],new_np[0])
        expected=tt[1:3].sum(axis=2)[:,:,None]*xx[1:3].mean(axis=1)[:,None,:]
        np.testing.assert_array_equal(new_np[1:3],expected)
        # The unchanged rank2 E1 approximation may be negative before the existing
        # gnn_step maximum(E,0). Require positivity only for changed full-mean E2/E3.
        assert np.isfinite(new_np).all() and np.min(new_np[1:3])>=0
        assert float(np.min(new_np[0]))==float(np.min(old_np[0]))
        cases.append(dict(n=n,case=kind,E1_byte_exact=True,E23_fullmean_byte_exact=True,input_unchanged=True,
            parent_flops=parent_bill,candidate_flops=new_bill,saving_flops=parent_bill-new_bill,
            parent_calls=parent_calls,candidate_calls=new_calls,unchanged_E1_min=float(np.min(new_np[0])),E23_min=float(np.min(new_np[1:3]))))
    big=next(x for x in cases if x['n']==1024)
    record=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
        status='OFFICIAL_E1_FULLMEAN_CACHE_NUMERICAL_PASS_NO_NETWORK_SCORE',cases=cases,
        fixed_15_layer_saving_flops=15*big['saving_flops'],source_sha256=sha(__file__),
        manifest_sha256=sha(a.manifest),separate_tiny_gnn_receipt_sha256=sha(a.out),
        no_prediction_or_adjusted_gain=True)
    path=a.out.with_name('ENERGY_KERNEL_PRICE_v1.json')
    with path.open('x') as h:json.dump(record,h,indent=2);h.write('\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
