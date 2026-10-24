"""Root-only fixed P4 fixtures, adapted from the frozen D5 probe.

CLI: NATIVE -B probe.py BUILD_SHA EXCLUSIVE_RESULT_JSON
Guard120s/7.5GB/zeroSwap; one BLAS thread. No estimator predictions or targets.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops

D=Path(__file__).resolve().parent
NATIVE=D.parent.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
PARENT_SHA='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')


def pin(path):
    p=Path(path).resolve();raw=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def checked(record):
    assert pin(record['path'])==record,record['path']


def summary(ctx):
    value=ctx.summary_dict()
    return dict(flops=int(value['flops_used']),operations=sum(
        int(v.get('calls',v.get('count',0))) for v in value['operations'].values()))


def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def check_sides(mod,layout,child):
    assert len(mod._PSDB_SIDES)==9 and mod._PACKED_ELEMENTS==391602176
    assert sum(len(mod.__dict__[name]) for name in mod._PB_CACHES)==682
    arena=np.asarray(mod._SCRATCH[0]);base=arena.__array_interface__['data'][0]
    assert arena.dtype==np.float32 and arena.size==391602176
    assert tuple(mod._PSDB_GROUPS)==((0,128,256),(128,128,128))
    for record in layout:
        key=tuple(record['key']);side=mod._PSDB_SIDES[key]
        depth=record['depth'] if child else record['parent_depth']
        kind,rows,cols=key;u=1<<depth;size=4*7**depth*(rows//u)*(cols//u)
        assert (side.kind,side.P,side.L,side.rows,side.cols)==(kind,4,depth,rows,cols)
        assert side._borrowed_arena is mod._SCRATCH[0]
        assert side._borrowed_start==record['start_elements']
        tree=np.asarray(side.T)
        assert tree.size==size and np.shares_memory(tree,arena)
        assert tree.__array_interface__['data'][0]-base==4*record['start_elements']
    if child:
        assert set(mod._PSD5_ACTIVE)=={tuple(r['key']) for r in layout if r['depth']==3}
        assert mod._PSD5_DEPTHS=={tuple(r['key']):r['depth'] for r in layout}


def main():
    assert len(sys.argv)==3
    target=Path(sys.argv[2]).resolve();assert not target.exists() and target.parent.is_dir()
    assert Path(sys.executable).absolute()==NATIVE and np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in THREADS)
    build_pin=pin(D/'build.json');assert build_pin['sha256']==sys.argv[1]
    build=json.loads((D/'build.json').read_text())
    for key in ('parent','candidate','parent_build','builder','protocol','fragment','primitive','pricing'):
        checked(build[key])
    assert build['primitive']==pin(__file__) and build['parent']['sha256']==PARENT_SHA
    parent=Path(build['parent']['path']).read_bytes();fragment=Path(build['fragment']['path']).read_bytes()
    candidate=Path(build['candidate']['path']).read_bytes()
    assert candidate==parent+b'\n'+fragment and candidate[:-build['suffix_bytes']]==parent
    assert build['append_only'] and build['source_inverse_exact']
    pricing=json.loads(Path(build['pricing']['path']).read_text())
    prices={r['depth']:r for r in pricing['fixed_price_table']}
    assert (prices[5]['helper_flops'],prices[3]['helper_flops'])==(707714016,738303136)
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
    sys.modules['whestbench']=stub
    modules={};setups={}
    for label in ('parent','child'):
        source=Path(build['parent' if label=='parent' else 'candidate']['path'])
        mod=types.ModuleType('psdb_depth3_'+label);mod.__file__=str(source);sys.modules[mod.__name__]=mod
        exec(compile(source.read_text(),str(source),'exec'),mod.__dict__)
        t0=time.monotonic()
        mod.Estimator().setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,
            api_version='2',scratch_dir=None,submission_dir=None,seed=0))
        elapsed=time.monotonic()-t0;assert 0<=elapsed<=4.1
        for key,count in (('bound',682),('lvfuse',649),('psdb_fused',9)):
            assert [r[1] for r in mod._PB_LOG if r[0]==key]==[count]
        check_sides(mod,build['side_layout'],label=='child')
        modules[label]=mod;setups[label]=dict(setup_s=elapsed,pb_log=list(mod._PB_LOG))
    records=[]
    for kind,seed in (('integer_gram',202609285),('gaussian_covariance',202609286)):
        rng=np.random.default_rng(seed);stage_one=None;parent_mod=modules['parent']
        if kind=='integer_gram':
            w=(2*rng.integers(0,2,size=(1024,1024))-1).astype(np.float32);a,b=w.T,w.copy()
        else:
            w=(rng.standard_normal((1024,1024))/32).astype(np.float32)
            h=(rng.standard_normal((1024,1024))/32).astype(np.float32);cov=h@h.T
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                cw=parent_mod._mm(parent_mod.fnp.asarray(cov),parent_mod.fnp.asarray(w))
            stage_one=summary(ctx);a,b=w.T,np.asarray(cw).copy();del h,cov,cw
        reference=a.astype(np.float64)@b.astype(np.float64)
        inputs=[hashlib.sha256(x.tobytes()).hexdigest() for x in (a,b)]
        runs=[];values={}
        for label,mod in modules.items():
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                aa,bb=mod.fnp.asarray(a),mod.fnp.asarray(b)
            conversions=summary(ctx);first_hash=None
            for rep in range(2):
                mod._TR_LAST[0]=None;t0,c0=time.monotonic(),time.process_time()
                with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                    out=mod._psdb_sym_upper(aa,bb)
                wall,cpu=time.monotonic()-t0,time.process_time()-c0;value=np.asarray(out).copy()
                assert value.shape==(1024,1024) and value.dtype==np.float32 and np.isfinite(value).all()
                digest=hashlib.sha256(value.tobytes()).hexdigest()
                if first_hash is None:first_hash=digest
                assert digest==first_hash
                relative=float(np.linalg.norm(value.astype(np.float64)-reference)/np.linalg.norm(reference))
                exact=bool(np.array_equal(value.astype(np.float64),reference))
                assert exact if kind=='integer_gram' else relative<=2e-4
                assert inputs==[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in (aa,bb)]
                assert inputs==[hashlib.sha256(x.tobytes()).hexdigest() for x in (a,b)]
                bill=summary(ctx);expected=(707714016,422) if label=='parent' else (738303136,378)
                assert (bill['flops'],bill['operations'])==expected,(label,bill)
                check_sides(mod,build['side_layout'],label=='child');assert mod._TR_LAST[0] is None
                runs.append(dict(arm=label,repeat=rep,wall_s=wall,cpu_s=cpu,**bill,
                    conversions=conversions,output_sha256=digest,relative_frobenius_error=relative,
                    integer_reference_exact=exact,repeat_bytes_exact=True))
                if rep==0:values[label]=value
                del out,value
            del aa,bb
        disagreement=float(np.linalg.norm(values['child'].astype(np.float64)-values['parent'])/np.linalg.norm(reference))
        records.append(dict(kind=kind,seed=seed,input_sha256=inputs,stage_one=stage_one,
            parent_child_relative_difference=disagreement,runs=runs))
        print(json.dumps(dict(event='case_complete',**records[-1])),flush=True)
        del a,b,w,reference,values
    assert 0<rss()<7500000000
    for mod in modules.values():
        mod._TR_LAST[0]=None;mod._pb_reset((None,None))
        assert mod._SCRATCH is None and not mod._PSDB_SIDES
        assert all(not mod.__dict__[name] for name in mod._PB_CACHES)
    for key in ('parent','candidate','parent_build','builder','protocol','fragment','primitive','pricing'):
        checked(build[key])
    checked(build_pin)
    result=dict(passed=True,candidate=build['candidate'],parent=build['parent'],build=build_pin,
        probe=pin(__file__),protocol=build['protocol'],pricing=build['pricing'],setups=setups,
        cases=records,process_peak_rss_bytes=rss(),all_repeat_bytes_exact=True,
        operand_immutability=True,arena_offsets_and_sharing_checked=True,side_pool_reset_passed=True,
        parent_byte_parity_required=False,rounding_note='D5-to-D3 changes FP32 grouping; fixed reference gates apply.',
        cpu_and_wall_diagnostic_only=True,external_guard_required=True,
        model_executed=False,network_targets_read=False,serving_qualified=False)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('D3_P4_FIXTURES_COMPLETE',flush=True)


if __name__=='__main__':main()
