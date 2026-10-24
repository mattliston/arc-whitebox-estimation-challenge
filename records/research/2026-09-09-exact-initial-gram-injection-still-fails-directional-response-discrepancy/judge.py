"""Read-only-array native bridge judge; no estimator/weights/truth execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BD = NEW/'birth_direction_fit_v1'
HY = NEW/'current_d50_hybrid_cp_qualification_v1'
CO = NEW/'current_bd_p_coadaptation_v1'
STORE = NEW.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
CASES = ('null','plus025','minus025','plus050','minus050')
VERIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
PANEL_SHA = '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade'
HELPER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
MODEL_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge',action='store_true')
    for flag in ('source','native','watch','build','hybrid','protocol'):
        parser.add_argument('--'+flag+'-sha256')
    args=parser.parse_args()
    if not args.judge:
        parser.print_help();return 0
    require(all(isinstance(getattr(args,k+'_sha256'),str) and len(getattr(args,k+'_sha256'))==64
                for k in ('source','native','watch','build','hybrid','protocol')),'Reviewed hashes required')
    out=HERE/'judgment_v1'
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    result=dict(schema='current_d50_native_p_bridge_judgment_v1',passed=False,completed=False,
                source_bindings={},input_bindings={},source_inverses={},native_records={},comparisons=[],secants=[],
                truth_accessed=False,weights_decoded=False,estimator_imported=False,card_calls=0,
                parameter_updates=0,fit_admitted=False,score_measured=False)
    write(out/'started.json',result)

    def pin(path,expected=None):
        path=Path(path).resolve()
        require(path.is_relative_to(NEW) or path.is_relative_to(NEW.parent/'data_gate')
                or path.is_relative_to(STORE),'Input path scope')
        digest=sha(path)
        require(expected is None or digest==expected,'Input SHA mismatch: '+str(path))
        result['input_bindings'][str(path)]=digest
        return digest

    def guard(*_):
        require(time.monotonic()-start<120,'Judge120s cap')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000,'Judge2GB cap')

    previous=signal.signal(signal.SIGALRM,guard)
    signal.setitimer(signal.ITIMER_REAL,.5,.5)
    try:
        require(Path(sys.prefix).resolve()==Path('/Users/matt/anaconda3'),'AP tiny-array judge')
        require(os.getpriority(os.PRIO_PROCESS,0)>=10 and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1',
                'nice10/no bytecode required')
        require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),'BLAS1 required')
        for path,digest in ((Path(__file__).resolve(),args.source_sha256),(HERE/'native.py',args.native_sha256),
            (HERE/'watch.py',args.watch_sha256),(HERE/'PROTOCOL.md',args.protocol_sha256),
            (HERE/'build_probes_v1/build.json',args.build_sha256),
            (HY/'stage_mps_gradient_v1/result.json',args.hybrid_sha256),
            (BD/'qualify_native_v2.py',VERIFIER_SHA),(CO/'qualify.py',HELPER_SHA),(CO/'model.py',MODEL_SHA)):
            result['source_bindings'][str(path)]=pin(path,digest)
        require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'Inert admission')
        native=module(HERE/'native.py','bridge_judge_native_admission')
        verifier=module(BD/'qualify_native_v2.py','bridge_frozen_native_verifier')
        verifier.verify_meter(pin)
        build=json.loads((HERE/'build_probes_v1/build.json').read_text())
        by_source={str(Path(x['path']).resolve()):x['sha256'] for x in build['candidates'].values()}

        def candidate_digest(path):
            require(str(path) in by_source,'Unknown fixed probe source')
            return by_source[str(path)]

        source=inspect.getsource(verifier.verify_native)
        edits=(
            ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
             "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)",
             '    expected_sha = candidate_digest(source)'),
            ('    expected_init = 1383913666 + (0 if source == BANK else 120)',
             '    expected_init = 1383913786'),
        )
        adapted=source
        for old,new in edits:
            require(adapted.count(old)==1,'Unexpected frozen verifier span')
            adapted=adapted.replace(old,new)
        inverse=adapted
        for old,new in reversed(edits):
            inverse=inverse.replace(new,old)
        require(inverse==source,'Verifier inverse differs')
        env=dict(verifier.__dict__,candidate_digest=candidate_digest)
        exec(compile(adapted,str(__file__)+':verify_native','exec'),env)
        result['source_inverses']['native_verifier_except_identity_and_init']=True
        result['source_inverses']['adapted_verifier_sha256']=hashlib.sha256(adapted.encode()).hexdigest()
        native_bytes={}
        for case in CASES:
            pins,admission=native.authenticate(case,args.native_sha256,args.build_sha256,args.hybrid_sha256,args.protocol_sha256)
            for path,digest in pins.items():pin(path,digest)
            folder=HERE/('run_'+case+'_v1')
            watch_folder=HERE/('watch_'+case+'_v1')
            wr, wc=watch_folder/'result.json',watch_folder/'completion.json'
            pin(wr);pin(wc)
            watched,seal=json.loads(wr.read_text()),json.loads(wc.read_text())
            require(watched['passed'] and watched['completed'] and watched['sources_unchanged']
                    and not watched['terminated'] and seal['passed'] and seal['completed']
                    and seal['result_sha256']==sha(wr),'Native external watchdog did not pass')
            require(watched['source_bindings'][str(HERE/'watch.py')]==args.watch_sha256
                    and watched['source_bindings'][str(HERE/'native.py')]==args.native_sha256
                    and watched['case']==case and watched['child_exit_code']==0,'Exact watched case')
            for path,digest in watched['source_bindings'].items():pin(path,digest)
            require(watched['elapsed_s']<900 and watched['maximum_sampled_rss_bytes']<8_000_000_000
                    and watched['maximum_swap_growth_bytes']<=2*1024**3,'Native external resources')
            acq=verifier.sealed(folder,watched['child_result_sha256'],pin)
            require(acq['source_bindings'][str(HERE/'native.py')]==args.native_sha256,'Actual native adapter')
            native_bytes[case]=env['verify_native'](acq,folder,(35,100),pin)
            result['native_records'][case]=dict(result_sha256=sha(folder/'result.json'),watch_sha256=sha(wr),
                    initialization_flops=acq['initialization_flops'],flops=[t['flops'] for t in acq['trials']],
                    complete_loop_verified=True)
        parent=verifier.sealed(BD/'panel_v2/terminal',PANEL_SHA,pin)
        for call,old in enumerate((parent['trials'][0],parent['trials'][1],parent['trials'][-1])):
            reference=verifier.payload(verifier.path(old['output']),old['output_bytes_sha256'],pin)
            require(native_bytes['null'][call]==reference,'Fresh d50 null differs from official-input native reference')
            require(result['native_records']['null']['flops'][call]==old['flops'],'Null complete bill changed')
        result['null_bytes_and_bills_exact']=True
        helper=module(CO/'qualify.py','bridge_original_scalar_helper')
        owner=module(CO/'model.py','bridge_original_scalar_owner')
        for path,digest in owner.authenticate().items():pin(path,digest)
        cpu_path=CO/'stage_cpu_forward_v1/result.json'
        pin(cpu_path,'6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607')
        cpu=json.loads(cpu_path.read_text())
        hybrid_path=HY/'stage_mps_forward_v1/result.json'
        pin(hybrid_path,'d6edae6b5828e09939d5afb7be7a3bc619f8e384fd909cbf105b26331a2943fc')
        hybrid=json.loads(hybrid_path.read_text())
        import numpy as np
        import torch
        require(np.__version__=='1.26.4' and torch.__version__=='2.7.0','Scalar numerical environment')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        losses={rank:{} for rank in (35,100)}
        for index,rank in enumerate((35,100)):
            cot_meta=cpu['cotangents'][str(rank)]
            pin(cot_meta['path'],cot_meta['sha256'])
            cot=helper.load_array(np,cot_meta)
            for case in CASES:
                value=np.frombuffer(native_bytes[case][index],dtype=np.float32).reshape(16,1024).copy()
                vals={}
                for label,rec in (('cpu',cpu),('hybrid',hybrid)):
                    meta=rec['outputs'][str(rank)][case]['prediction']
                    pin(meta['path'],meta['sha256'])
                    vals[label]=helper.load_array(np,meta)
                maximum=float(np.max(np.abs(value.astype(np.float64)-vals['hybrid'])))
                cpu_max=float(np.max(np.abs(value.astype(np.float64)-vals['cpu'])))
                loss=float(helper.scalar(torch,owner,torch.from_numpy(value),cot,torch.float64))
                losses[rank][case]=loss
                item=dict(rank=rank,case=case,native_hybrid_maximum_absolute_difference=maximum,
                          native_cpu_maximum_absolute_difference=cpu_max,synthetic_loss=loss)
                result['comparisons'].append(item)
                require(math.isfinite(loss) and maximum<=1e-5,'Native/hybrid all16 gate')
            for suffix,amplitude in (('025',.025),('050',.05)):
                secant=(losses[rank]['plus'+suffix]-losses[rank]['minus'+suffix])/(2*amplitude)
                other=next(x for x in hybrid['secants'] if x['rank']==rank and x['amplitude']==amplitude)['symmetric_secant']
                old=next(x for x in cpu['secants'] if x['rank']==rank and x['amplitude']==amplitude)['symmetric_secant']
                relative=abs(secant-other)/max(abs(secant),abs(other),1e-10)
                item=dict(rank=rank,amplitude=amplitude,native_secant=secant,hybrid_secant=other,cpu_secant=old,
                          relative_hybrid_error=relative,relative_cpu_error=abs(secant-old)/max(abs(secant),abs(old),1e-10))
                result['secants'].append(item)
                require(math.isfinite(secant) and relative<=.05,'Native/hybrid unchanged5% scalar gate')
        for path,digest in result['input_bindings'].items():require(sha(path)==digest,'Bound judge input changed')
        require('flopscope' not in sys.modules and 'threadpoolctl' not in sys.modules,'No estimator/numerical runtime')
        guard()
        result.update(passed=True,completed=True,sources_unchanged=True,native_bridge_passed=True)
    except BaseException:
        result.update(passed=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result.update(elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','error')},sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
