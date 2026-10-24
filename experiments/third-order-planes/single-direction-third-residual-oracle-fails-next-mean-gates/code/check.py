"""Inert fixed q1 residual capacity test in the preserved old bridge frame."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
OFFENV=NEW.parent/'data_gate/.venv'
OUT=HERE/'result_v1'
BLAS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
      'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
PINS={
 'residual_one_high_oracle_v1.py':'22e9effb76bc59ccffaa6d41444524faf928b32741a4e2b5df4d3ce432a21ddc',
 'current_pair_oracle_math_v1.py':'ed1bc16bdf6b1b2467c889bd5c23e1fe849061feef8515140ee9a8b1801dc4bb',
 'quadratic_skew_covariance_v1.py':'16c120afe013168941c72d024ab18951f11e3424eeb22fc3974e0f19c1669fa3',
 'radial_mehler.py':'828e7e7dac78437bc528ec58b1235ea96cae92b7bcb1e22dcedde977808b6495',
 'radial_binary_compact.py':'9759750857aab404da2774d336d809b0933a57fb4bdf1fe64c297fe7a1fce108',
 'one_high_actual_value_r35_l12_v3.json':'b49af328f00333e058954c464b8ad21dfae1bd5399a066837b39b6b0893ef569',
 'one_high_actual_value_r35_l12_v3.npz':'a9fb6d3a61a75497f5d905da8a6e38eb639805b53a9d76ebbe9a141c28a8b8f9',
 'current_pair_oracle_attribution_v2/inputs_rank_0035.npz':'de8a8a8e096e8b9d060350d41f04b5b50fce88f678d76040d94710813a77bc85',
 'rows_covariance_innovation_probe_v2/row_0035/data.npz':'893b1639ce5cb2a28c604aec3e642e5ced072c7e7402384ed97476bb7443cb4b',
 'residual_one_high_inputs_r35_l12_v1.npz':'512c1bbbc3498d1cf02a7b7ad9159416b75a41a8fba0363d20b8e9c40c463efd',
 'residual_one_high_truth_r35_l12_A_v1.npz':'23460f0205733e56a37880ebcfab824b20986acc1067f0645eaa03dfe332ad66',
}

def require(value,message):
    if not value: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def bind(path):
    path=Path(path).resolve();s=path.stat();digest=sha(path);t=path.stat()
    require((s.st_size,s.st_mtime_ns)==(t.st_size,t.st_mtime_ns),'File changed during hash')
    return dict(path=str(path),sha256=digest,bytes=t.st_size,mtime_ns=t.st_mtime_ns)

def write(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return sha(path)

def read(path): return json.loads(Path(path).read_text())

def peak():
    value=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform=='darwin' else value*1024

def guard(start):
    require(peak()<1_000_000_000 and time.perf_counter()-start<60,'Fixed resource gate')

def functions(np):
    env={'np':np}
    for name,names in (
      ('radial_mehler.py',('derivatives',)),
      ('quadratic_skew_covariance_v1.py',('response',)),
      ('current_pair_oracle_math_v1.py',('mean_change','exact_change')),
      ('residual_one_high_oracle_v1.py',('plane','repeated_push','distinct_plane')),
      ('radial_binary_compact.py',('select',))):
        tree=ast.parse((NEW/name).read_text())
        found=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        require({node.name for node in found}==set(names),'Missing pure function')
        exec(compile(ast.Module(body=found,type_ignores=[]),str(NEW/name),'exec'),env)
    return env

def fixture(np,env):
    rng=np.random.default_rng(9081301);n=6
    raw=rng.normal(size=(n,n,n))
    T=sum(raw.transpose(p) for p in itertools.permutations(range(3)))/6
    u=np.array([1.,2.,-1.,0.,1.,-2.])/math.sqrt(11.)
    W=rng.normal(size=(n,n))/math.sqrt(n)
    M=np.einsum('i,ijk->jk',u,T);h=M@u;c=float(u@h)
    B=M-(np.outer(u,h)+np.outer(h,u))/2+c*np.outer(u,u)/3
    represented=np.einsum('i,jk->ijk',u,B)+np.einsum('j,ik->ijk',u,B)+np.einsum('k,ij->ijk',u,B)
    Q=np.eye(n)-np.outer(u,u)
    projected=T-np.einsum('ia,jb,kc,abc->ijk',Q,Q,Q,T,optimize=True)
    i,j,k=np.indices((n,n,n));distinct=projected*((i!=j)&(i!=k)&(j!=k))
    expected=np.einsum('abc,ai,bi,cj->ij',distinct,W,W,W,optimize=True)
    actual=env['distinct_plane'](M[None],u[:,None],W,np.arange(n))
    errors=dict(compact_tensor=float(np.max(np.abs(represented-projected))),
                distinct_push=float(np.max(np.abs(actual-expected))))
    require(max(errors.values())<5e-13,'Fixed n6/q1 dense fixture')
    return dict(passed=True,errors=errors)

def execute():
    require(Path(sys.prefix).resolve()==OFFENV.resolve(),'Pinned OFFPY required')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    require(all(os.environ.get(k)=='1' for k in BLAS) and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','BLAS1/no bytecode')
    require(not OUT.exists(),'Refusing overwrite')
    OUT.mkdir(exist_ok=False);start=time.perf_counter()
    result=dict(passed=False,completed=False,numerical_qualified=False,rank=35,
        source_layer=12,pair_layer=13,value_layer=14,basis_column=3,q=1,
        started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        final_raw_claim=False,complete_cost_claim=False,current409f_claim=False,
        privileged_source=True,target_access=False,artifacts={})
    sources={}
    def alarm(sig,frame): raise TimeoutError('Fixed60s q1 information cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(60)
    try:
        for name in ('residual_one_high_inputs_r35_l12_v1.json',
                     'residual_one_high_truth_r35_l12_A_v1.json',
                     'residual_one_high_truth_r35_l12_B_v1.json'):
            item=bind(NEW/name);sources[item['path']]=item
        prereq=read(NEW/'residual_one_high_inputs_r35_l12_v1.json')
        require(prereq['original_covariance_target_byte_exact'] and prereq['parent_plane_max_difference']==0 and prereq['q']==4,'Original capture qualification')
        pins=dict(PINS)
        for stream in ('A','B'):
            rec=read(NEW/f'residual_one_high_truth_r35_l12_{stream}_v1.json')
            require(rec['small_certificate']['passed'] and rec['samples']==8_000_000
                and rec['frozen_input_sha256']==PINS['residual_one_high_inputs_r35_l12_v1.npz'],'Stream provenance')
            name=f'residual_one_high_truth_r35_l12_{stream}_v1.npz'
            require(name not in pins or pins[name]==rec['output_sha256'],'Conflicting stream hash')
            pins[name]=rec['output_sha256']
        for name,digest in pins.items():
            item=bind(NEW/name);require(item['sha256']==digest,'Frozen source/input: '+name)
            sources[item['path']]=item
        for path in (Path(__file__).resolve(),HERE/'PROTOCOL.md'):
            item=bind(path);sources[item['path']]=item
        result['sources']=sources
        result['versions']={k:importlib.metadata.version(k) for k in ('numpy','threadpoolctl')}
        require(result['versions']=={'numpy':'2.4.6','threadpoolctl':'3.6.0'},'Pinned versions')
        write(OUT/'started.json',result)
        import numpy as np
        from threadpoolctl import threadpool_info,threadpool_limits
        np.seterr(over='raise',divide='raise',invalid='raise',under='ignore')
        with threadpool_limits(limits=1):
            result['threadpools']=threadpool_info()
            require(all(p['num_threads']==1 for p in result['threadpools'] if p.get('user_api')=='blas'),'Reported BLAS pool')
            result['thread_attestation']='Explicit BLAS1; empty report not independently observed count.'
            env=functions(np);result['fixture']=fixture(np,env);write(OUT/'fixture.json',result['fixture'])
            with np.load(NEW/'residual_one_high_inputs_r35_l12_v1.npz',allow_pickle=False) as z:
                U=z['U'];model=z['model_slices'];W=z['next_weight'];eigenvalues=z['eigenvalues']
            require(U.shape==(1024,4) and model.shape==(4,1024,1024) and W.shape==(1024,1024),'Fixed shapes')
            require(np.all(np.diff(eigenvalues)>=0) and np.max(np.abs(U.T@U-np.eye(4)))<1e-10,'Top basis ordering/orthogonality')
            with np.load(NEW/'current_pair_oracle_attribution_v2/inputs_rank_0035.npz',allow_pickle=False) as z:
                mu,C,host=(z[k][5].astype(np.float64) for k in ('mu','Cpre','host'))
            sigma=np.sqrt(np.diag(C));alpha=mu/sigma
            pdf=np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
            g=np.asarray([.5*math.erfc(-float(a)/math.sqrt(2)) for a in alpha])
            selection_alpha=mu/np.sqrt(np.maximum(np.diag(C),1e-10))
            selection_gate=np.asarray([.5*math.erfc(-float(a)/math.sqrt(2)) for a in selection_alpha])
            _,alive=env['select'](selection_gate,.01);live=alive[:,None]*alive[None,:];np.fill_diagonal(live,True)
            common=(C/np.outer(sigma,sigma),sigma,alpha,pdf,g,pdf+alpha*g,host/sigma)
            deltas={}
            for stream in ('A','B'):
                with np.load(NEW/f'residual_one_high_truth_r35_l12_{stream}_v1.npz',allow_pickle=False) as z:
                    M=z['slices'][3:4]-model[3:4]
                delta=env['distinct_plane'](M,U[:,3:4],W,np.arange(1024))*live
                require(np.isfinite(delta).all(),'Nonfinite q1 action')
                deltas[stream]=delta;guard(start)
            with (OUT/'actions_sealed.npz').open('xb') as f:
                np.savez(f,delta_A=deltas['A'],delta_B=deltas['B'],u=U[:,3],live=live)
            result['artifacts']['actions_sealed.npz']=bind(OUT/'actions_sealed.npz')
            write(OUT/'action_seal.json',dict(utc=datetime.now(timezone.utc).isoformat(),
                covariance_target_decoded=False,privileged_slices_decoded=True,
                artifact=result['artifacts']['actions_sealed.npz'],sources=sources))
            result['artifacts']['action_seal.json']=bind(OUT/'action_seal.json')
            result['target_access']=True
            with np.load(NEW/'rows_covariance_innovation_probe_v2/row_0035/data.npz',allow_pickle=False) as z:
                E,wn,resp=(z[k][5].astype(np.float64) for k in ('target','weight','response'))
            require(E.shape==wn.shape==(1024,1024) and resp.shape==(1024,),'Target consumer shapes')
            value=lambda mat:np.sum(wn*(mat@wn),axis=0)*resp
            baseline=value(E);ce=float(np.sum(E*E));ve=float(baseline@baseline)
            require(math.isfinite(ce) and math.isfinite(ve) and ce>0 and ve>0,'Baseline energies')
            def measure(delta):
                dm,dc=env['exact_change'](common,dS=delta/(sigma[:,None]**2*sigma[None,:]))
                dv=value(dc)
                rec=dict(covariance_error_ratio=float(np.sum((E-dc)**2)/ce),
                    next_mean_error_ratio=float(np.sum((baseline-dv)**2)/ve),
                    correction_cosine=float(baseline@dv/math.sqrt(ve*float(dv@dv))),
                    correction_relative_norm=float(np.linalg.norm(dv)/math.sqrt(ve)))
                require(np.isfinite(dm).all() and np.isfinite(dc).all() and all(math.isfinite(v) for v in rec.values()),'Finite response')
                return rec,dc,dv
            old=read(NEW/'one_high_actual_value_r35_l12_v3.json')
            require(all(old['byte_exact_boundary'].values()),'Existing boundary qualifier')
            replays={}
            with np.load(NEW/'one_high_actual_value_r35_l12_v3.npz',allow_pickle=False) as z:
                for stream in ('A','B'):
                    arm='oracle_'+stream;rec,dc,dv=measure(z[arm+'_delta_K21'])
                    differences={k:abs(rec[k]-old['results'][arm][k]) for k in ('covariance_error_ratio','next_mean_error_ratio')}
                    for name,actual in (('delta_cov',dc),('delta_value',dv)):
                        expected=z[arm+'_'+name]
                        differences[name+'_relative']=float(np.linalg.norm(actual-expected)/max(float(np.linalg.norm(expected)),1e-300))
                    require(max(differences.values())<=1e-10,'Old q4 response replay')
                    replays[arm]=differences
            result['old_q4_replay']=replays;result['numerical_qualified']=True
            result['baseline_covariance_mse']=ce/(1024**2)
            result['baseline_next_mean_response_mse']=ve/1024
            result['results']={}
            for stream in ('A','B'):
                rec,dc,dv=measure(deltas[stream]);result['results'][stream]=rec
                with (OUT/f'response_{stream}.npz').open('xb') as f: np.savez(f,delta_cov=dc,delta_value=dv)
                result['artifacts'][f'response_{stream}.npz']=bind(OUT/f'response_{stream}.npz')
                print(json.dumps(dict(stream=stream,**rec)),flush=True);guard(start)
            result['passed']=all(v['next_mean_error_ratio']<=.85 and v['covariance_error_ratio']<=1 for v in result['results'].values())
            result['completed']=True
        require(not any(k=='torch' or k.startswith(('torch.','scipy')) for k in sys.modules),'Forbidden package')
        for item in (*sources.values(),*result['artifacts'].values()):
            require(bind(item['path'])==item,'Changed input/source/output')
        result['hashes_unchanged']=True;guard(start)
    except BaseException:
        result['passed']=False;result['error']=traceback.format_exc()
    finally: signal.alarm(0)
    result.update(elapsed_s=time.perf_counter()-start,peak_rss_bytes=peak(),finished_utc=datetime.now(timezone.utc).isoformat())
    digest=write(OUT/'result.json',result)
    write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=digest))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','numerical_qualified','elapsed_s','peak_rss_bytes','results','error')}),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute: raise SystemExit(execute())
    parser.print_help()
