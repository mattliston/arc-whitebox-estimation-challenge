"""Fixed one-product native-reference/NumPy-free-client qualification driver.

Inert without --run. Numerical fixtures are synthetic, never MLPs or targets.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
GATE=NEW.parent/'data_gate'
OUT=HERE/'run_v1'
SOURCE=NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'
ENV=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
    MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
PINS={
    SOURCE:'d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451',
    NEW/'packed_arena_generic.py':'4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e',
    NEW/'current_d50_d6_cost_census_v1/price.json':'995bd57f77d526cee191ec407473946c1d3116ed7390754ba635485458c76c72',
    NEW/'current_d50_d6_cost_census_v1/native/result.json':'d3d080115b8b27ad5af26196d045c3324db2c5c44496357b2ab8f4ec658666c2',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def stop_owned(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=5)


def run_role(role,contract_sha):
    scratch=Path(tempfile.mkdtemp(prefix='arc_p2_d6_product_',dir='/private/tmp'))
    url=f'ipc://{scratch}/flops.sock'
    env=dict(os.environ,**ENV,FLOPSCOPE_SERVER_URL=url)
    server=client=None
    samples=[]
    started=time.monotonic()
    record=dict(role=role,passed=False,peak_combined_rss_bytes=0,samples=samples,
        memory_scope='200ms sampled combined client/server RSS, not continuous/cgroup peak')
    try:
        with (OUT/f'{role}_server.log').open('x') as log:
            server=subprocess.Popen([str(NEW/'openblas_runtime_env_v1/bin/python'),'-m',
                'flopscope_server','--url',url,'--timeout','150'],env=env,stdout=log,stderr=subprocess.STDOUT)
            for _ in range(100):
                assert server.poll() is None,'Isolated server exited'
                if (scratch/'flops.sock').exists():
                    break
                time.sleep(.05)
            assert (scratch/'flops.sock').exists(),'Server did not become ready'
            client=subprocess.Popen([str(GATE/'.client-venv/bin/python'),'-B',str(HERE/'client.py'),
                '--role',role,'--contract-sha256',contract_sha],env=env,cwd=HERE)
            record.update(server_pid=server.pid,client_pid=client.pid)
            while client.poll() is None:
                elapsed=time.monotonic()-started
                assert elapsed<120,'Fixed role wall limit'
                assert server.poll() is None,'Server exited before client'
                sample=subprocess.run(['/bin/ps','-o','rss=','-p',f'{server.pid},{client.pid}'],
                    capture_output=True,text=True,check=True)
                rss=sum(int(x) for x in sample.stdout.split())*1024
                samples.append([elapsed,rss])
                record['peak_combined_rss_bytes']=max(record['peak_combined_rss_bytes'],rss)
                assert rss<8e9,'Fixed8GB combined memory guard'
                time.sleep(.2)
            record['client_exit_code']=client.returncode
            assert client.returncode==0,'Product client failed'
            record['passed']=True
    except BaseException:
        record['error']=traceback.format_exc()
    finally:
        stop_owned(client);stop_owned(server)
        record['wall_s']=time.monotonic()-started
        record['max_single_child_rss_bytes_process_lifetime']=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if record['wall_s']>=120 or record['max_single_child_rss_bytes_process_lifetime']>=8e9:
            record['passed']=False
            record['final_resource_gate_error']='Final role wall or single-child peak exceeded the fixed guard'
        write(OUT/f'{role}_resources.json',record)
    assert record['passed'],record.get('error')
    return record


def judge(np,kernel,resources):
    roles={}
    reference=np.load(OUT/'reference_f64.npy',allow_pickle=False)
    for role in ('d4','breadth_d6','serial_d6'):
        path=OUT/role/'result.json'
        result=json.loads(path.read_text())
        seal=json.loads((path.parent/'completion.json').read_text())
        assert result['passed'] and result['completed'] and seal['passed'] and seal['result_sha256']==sha(path)
        assert len(result['trials'])==3 and resources[role]['passed']
        assert 0<result['setup']['wall_time_s']<30
        expected=(dict(add=28,subtract=32,stack=8,copyto=4,matmul=1) if role=='d4' else
                  dict(add=42,subtract=48,stack=12,copyto=6,matmul=1) if role=='breadth_d6' else kernel.WARM_PRIMITIVE_CALLS)
        depth=4 if role=='d4' else 6
        for call in result['trials']:
            summary=call['summary']
            assert 0<summary['wall_time_s']<30
            assert 0<call['verification']['wall_time_s']<30
            assert all(math.isfinite(v) and v>=0 for k,v in summary.items()
                if ('time' in k or k=='flops_used') and isinstance(v,(float,int)))
            assert sha(call['output'])==call['output_sha256']
            if call['call']:
                assert summary['flops_used']==kernel.price(1024,8192,1024,depth)
                assert all(summary['operations'][k]['calls']==v for k,v in expected.items())
            array=np.fromfile(call['output'],dtype=np.float32).reshape(1024,1024)
            assert np.isfinite(array).all()
            error=array.astype(np.float64)-reference
            call['fp64_reference_error']=dict(rms=float(np.sqrt(np.mean(error*error))),maximum=float(np.max(np.abs(error))))
        digests=[t['output_sha256'] for t in result['trials']]
        assert len(set(digests))==1,'Own repeat changed'
        roles[role]=result
    assert roles['serial_d6']['trials'][0]['output_sha256']==roles['breadth_d6']['trials'][0]['output_sha256'],'Serial and breadthD6 differ in bytes'
    extra=roles['serial_d6']['setup']['flops_used']-roles['d4']['setup']['flops_used']
    assert extra==561056000,'Additional setup differs from complete-cost census'
    d4_time=max(t['summary']['wall_time_s'] for t in roles['d4']['trials'][1:])
    serial_time=max(t['summary']['wall_time_s'] for t in roles['serial_d6']['trials'][1:])
    ratio=serial_time/d4_time
    residual=max(t['summary']['residual_wall_time_s'] for t in roles['serial_d6']['trials'])
    timing_gate=ratio<=1.35 and residual<=.20
    write(OUT/'judged_roles.json',roles)
    return dict(numerical_and_billing_gate_passed=True,timing_gate_passed=timing_gate,
        single_product_qualification_passed=timing_gate,serial_breadth_bytes_equal=True,
        d4_max_warm_context_wall_s=d4_time,serial_max_warm_context_wall_s=serial_time,
        serial_to_d4_warm_wall_ratio=ratio,max_serial_residual_s=residual,
        additional_serial_setup_flops=extra,whole_estimator_qualified=False,score_measured=False,
        role_results={role:dict(result_sha256=sha(OUT/role/'result.json'),
            fp64_reference_error=r['trials'][0]['fp64_reference_error']) for role,r in roles.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--kernel-sha256')
    parser.add_argument('--client-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    assert sha(__file__)==args.source_sha256
    assert sha(HERE/'kernel.py')==args.kernel_sha256 and sha(HERE/'client.py')==args.client_sha256
    assert Path(sys.prefix).resolve()==(GATE/'.venv').resolve()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert all(os.environ.get(k)==v for k,v in ENV.items())
    pins={str(p):sha(p) for p in (*PINS,Path(__file__),HERE/'kernel.py',HERE/'client.py',HERE/'PROTOCOL.md')}
    for path,digest in PINS.items():
        assert pins[str(path)]==digest,path
    assert json.loads((NEW/'current_d50_d6_cost_census_v1/price.json').read_text())['advance_to_single_product_screen']
    import kernel
    audit=kernel.audit_sources(SOURCE.read_text(),(NEW/'packed_arena_generic.py').read_text(),(HERE/'kernel.py').read_text())
    OUT.mkdir()
    result=dict(passed=False,completed=False,truth_accessed=False,score_measured=False,
        started_utc=datetime.now(timezone.utc).isoformat(),source_bindings=pins,resources={})
    write(OUT/'started.json',result)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info,threadpool_limits
        assert importlib.metadata.version('numpy')=='2.4.6'
        assert importlib.metadata.version('flopscope')=='0.12.0'
        with threadpool_limits(limits=1):
            result['driver_threadpools']=threadpool_info()
            assert all(p['num_threads']==1 for p in result['driver_threadpools'] if p.get('user_api')=='blas')
            rng=np.random.default_rng(9309094)
            left=rng.standard_normal((1024,8192),dtype=np.float32)/np.float32(math.sqrt(8192))
            right=rng.standard_normal((8192,1024),dtype=np.float32)
            reference=left.astype(np.float64)@right.astype(np.float64)
            for name,array in (('left.npy',left),('right.npy',right),('reference_f64.npy',reference)):
                with (OUT/name).open('xb') as stream:
                    np.save(stream,array)
            del left,right,reference
            gc.collect()
            inputs={str(OUT/name):sha(OUT/name) for name in ('left.npy','right.npy','reference_f64.npy')}
            contract=dict(source_bindings=pins,input_bindings=inputs,source_audit=audit,env=ENV,
                shape=[1024,8192,1024],seed=9309094,truth_accessed=False)
            write(OUT/'contract.json',contract)
            digest=sha(OUT/'contract.json')
            for role in ('d4','breadth_d6','serial_d6'):
                result['resources'][role]=run_role(role,digest)
            result.update(judge(np,kernel,result['resources']))
            assert all(sha(p)==v for p,v in pins.items()) and all(sha(p)==v for p,v in inputs.items())
        result.update(passed=True,completed=True)
    except BaseException:
        result['error']=traceback.format_exc()
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    write(OUT/'result.json',result)
    write(OUT/'completion.json',dict(passed=result['passed'],result_sha256=sha(OUT/'result.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_bindings','resources','driver_threadpools')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
