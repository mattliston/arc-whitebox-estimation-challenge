"""Target-free scalar readout of the five already-saved parity pairs. No gate override."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CANDIDATE = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
MAPPING = ((0,0,35),(1,1,100),(2,7,500),(3,8,3016500),(4,14,35))
CHAINS = {
    'native_panel_v1': ('ceae3540a59b10f9ebf90885ff2b9b70f8154fc50a6cfe22e29ad5ff01d4e911',
        '8f31681349242356655de980eeaa44d3401812e941d99f5202b145dce8bb2983',
        '1ff6639648aca4e8fb60793751bcc7b281c97376d0f2d10b500bdb475d1630fc'),
    'runtime_v1': ('c8d325e35229396bf67f63fd65e92f09f75ed8305ef1e169388d975507318af0',
        'b37e54981a24af7433fd65b8ca90b38fb84d092a910f91c7655542b658169f6d',
        '40b6bde559cbee4960cd8a5202bd12c08e241cfa3d5dcea38d460b43a67bcc48'),
    'fidelity_v2': ('fcffdc67d39025333f5cb1c15cd1ee5cbd2c1df42cec01355b93d67ce0c381c9',
        'ec07d8e9b44dad72b36d96a589068e4f248cc7495febfd6cbbc1e5d43af79354',
        '0281a990041ee93dd1925fefbd51a4dde216409606789e5822b156fe4e511a0a'),
    'fidelity_watch_v2': ('1dccf7ea43149a8f19d8b5fc79bddb93a9ff1d88bf3a555b42d87159e20100f3',
        '107fa0e8cbe1b3a646e92b15e707b6b762ab0e4742472cabf682777d335f15c0',
        'b6ec1d725247d61c553531585076175aabbefbbb59ac2b9276f89a6281c772e6'),
}
CAP, RSS = 120., 1_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def payload(data, native):
    if native:
        require(data[:8] in (b'\x93NUMPY\x01\x00',b'\x93NUMPY\x02\x00'), 'Standard NPY header')
        size = 2 if data[6] == 1 else 4
        offset = 8+size
        length = int.from_bytes(data[8:offset], 'little')
        require(length < 2048, 'Bounded scalar NPY header')
        header = ast.literal_eval(data[offset:offset+length].decode('latin1'))
        require(header == {'descr':'<f4','fortran_order':False,'shape':(16,1024)}, 'Exact C-order FP32 output')
        data = data[offset+length:]
    require(len(data) == 16*1024*4, 'Exact output size')
    values = struct.unpack('<16384f',data)
    require(all(math.isfinite(x) and x >= 0 for x in values), 'Finite nonnegative all16')
    return values, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256',required=True)
    args = parser.parse_args()
    if not args.run:
        return 0
    began = previous = time.monotonic()
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.dont_write_bytecode
        and sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and all(os.environ.get(k) == '1' for k in BLAS), 'Original exclusive OFFPY/nice10/sixBLAS1')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == args.source_sha256, 'Reviewed diagnostic source')
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA, 'Original monitor transport')
    spec = importlib.util.spec_from_file_location('saved_parity_transport',TRANSPORT)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    mon,classifier = h.transport()
    out = HERE/'saved_parity_diagnostic_v1'; out.mkdir(exist_ok=False)
    result = dict(schema='shared_saved_parity_diagnostic_v1',passed=False,completed=False,
        diagnostic_only=True,production_fidelity_qualified=False,submission_authorized=False,
        truth_accessed=False,truth_bindings={},estimator_executed=False,full_model_calls=0,
        parameter_updates=0,numerical_libraries_imported=False,source_bindings={str(Path(__file__)):args.source_sha256,
        str(TRANSPORT):TRANSPORT_SHA,**{str(p):v for p,v in h.MON_PINS.items()}},
        input_bindings={},checks=[],samples=[],maximum_combined_rss_bytes=0,
        maximum_swap_growth_bytes=0,maximum_interval_s=0.,original_tolerance=1e-5)
    swap0 = None
    def guard(*unused):
        require(time.monotonic()-began < CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < RSS,
            'Original120s/1GB diagnostic cap')
    def observe():
        nonlocal previous,swap0
        guard(); absolute = min(previous+1.,began+CAP)
        snapshot = mon.collect(absolute)
        scope = mon.scope(snapshot['rows'],os.getpid(),None)
        require(not classifier(snapshot['rows'],scope.pop('allowed')), 'No foreign numerical worker')
        if swap0 is None:
            swap0 = snapshot['swap']['used_bytes']
        growth = snapshot['swap']['used_bytes']-swap0
        require(growth <= 0 and scope['combined_rss_bytes'] < RSS and snapshot['commands_reaped']
            and not mon.PENDING, 'No added swap; full tree1GB; observation groups retired')
        result['samples'].append(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            transport=snapshot['command_records']))
        previous,interval = mon.complete(previous,absolute)
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
    def tick():
        guard()
        if time.monotonic()-previous >= .15:
            observe()
    def read(path,digest):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2)
            and path.stat().st_size < 16*1024**2 and 'truth' not in path.name.lower()
            and 'truthpost' not in path.parts, 'Only bounded explicit target-free input')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, 'Exact saved bytes '+str(path))
        result['input_bindings'][str(path)] = digest; tick()
        return data
    signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        observe()
        packets = {}
        for folder,digests in CHAINS.items():
            r,c,f = [json.loads(read(HERE/folder/name,digest)) for name,digest in
                zip(('result.json','completion.json','finalization.json'),digests)]
            require(c['result_sha256'] == f['result_sha256'] == digests[0]
                and f['completion_sha256'] == digests[1], 'Actual three-seal linkage')
            expected = folder in ('native_panel_v1','runtime_v1')
            require(all(x['passed'] is expected and x['completed'] is expected for x in (r,c,f)),
                'Preserve actual positive acquisition and negative fidelity outcomes')
            packets[folder] = r
        n,p,negative = (packets[k] for k in ('native_panel_v1','runtime_v1','fidelity_v2'))
        require(n['candidate_sha256'] == p['candidate_sha256'] == negative['candidate_sha256'] == CANDIDATE
            and negative['truth_accessed'] is False and not negative['truth_bindings']
            and 'Original all16 vector native/client tolerance<=1e-5' in negative['error'], 'Same actual failed parity candidate')
        read(HERE/'build_v1/estimator.py',CANDIDATE)
        for prod,call,rank in MAPPING:
            nt,pt = n['trials'][call],p['observed_records'][prod]
            require(nt['rank'] == pt['rank'] == rank and nt['call'] == call and pt['index'] == prod,
                'Exact fixed output mapping')
            require(p['fixture_bindings'][str(rank)] == {k:n['fixture_bindings'][f'weights_{rank}'][k]
                for k in ('path','sha256')}, 'Same exact physical fixture')
            local,local_sha = payload(read(nt['output'],nt['output_sha256']),True)
            remote,remote_sha = payload(read(NEW/f"{p['tag']}_{prod}_r{rank}.f32",pt['output_sha256']),False)
            require(local_sha == nt['output_bytes_sha256'] and remote_sha == pt['output_sha256'], 'Exact payload hashes')
            layers = []
            for layer in range(16):
                delta = [remote[i]-local[i] for i in range(layer*1024,(layer+1)*1024)]
                location = max(range(1024),key=lambda i:abs(delta[i]))
                layers.append(dict(layer=layer,max_abs=max(map(abs,delta)),argmax=location,
                    signed_mean=math.fsum(delta)/1024,mse_between_predictions=math.fsum(x*x for x in delta)/1024,
                    above_original_tolerance=sum(abs(x)>1e-5 for x in delta)))
                tick()
            result['checks'].append(dict(production_call=prod,native_call=call,rank=rank,layers=layers,
                max_abs=max(row['max_abs'] for row in layers),native_flops=nt['flops'],production_flops=pt['flops'],
                native_payload_sha256=local_sha,production_payload_sha256=remote_sha,
                own_complete_flops_delta=pt['flops']-nt['flops']))
        for path,digest in list(result['input_bindings'].items()):
            read(path,digest)
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Pure saved scalar diagnostic')
        observe()
        result.update(passed=True,completed=True,hashes_unchanged=True,
            all_five_pairs_diagnosed=True,all_five_original_vector_gates_passed=all(x['max_abs']<=1e-5 for x in result['checks']))
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        mon.retire_pending(time.monotonic()+.5)
        result.update(elapsed_s=time.monotonic()-began,initial_swap_bytes=swap0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            all_observation_groups_retired=not mon.PENDING,actual_completed_samples=True)
        ok = result['passed'] and time.monotonic()-began<CAP and time.monotonic()-previous<=1.
        result['passed'] = result['completed'] = bool(ok)
        h.write(out/'result.json',result)
        rh = hashlib.sha256((out/'result.json').read_bytes()).hexdigest()
        h.write(out/'completion.json',dict(passed=bool(ok),completed=bool(ok),result_sha256=rh))
        ch = hashlib.sha256((out/'completion.json').read_bytes()).hexdigest()
        ok = ok and time.monotonic()-began<CAP and time.monotonic()-previous<=1.
        h.write(out/'finalization.json',dict(passed=bool(ok),completed=bool(ok),result_sha256=rh,
            completion_sha256=ch,elapsed_s=time.monotonic()-began,publication_interval_s=time.monotonic()-previous))
    print(json.dumps({k:result.get(k) for k in ('passed','elapsed_s','maximum_combined_rss_bytes',
        'maximum_interval_s','maximum_swap_growth_bytes','all_five_original_vector_gates_passed','checks','error')},allow_nan=False))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
