"""Fixed no-fit development acquisition, then separate sealed truth scoring."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
MODEL = NEW/'cp_free_donly_objective_v1'
BASELINE = NEW/'unused_materializations_v1/native'
HELPER = NEW/'check_response_rows_adjusted_v1.py'
HELPER_SHA = '8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304'
BASELINE_SHA = 'a9ccecaa3c0ca79f070af3c0961a6a952f08f4d9712e82a3d6af4d9a0d1d0951'
BASELINE_SOURCE = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
VERSIONS = dict(flopscope='0.12.0',whestbench='0.16.0',numpy='2.4.6',threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
RSS_CAP = 4_000_000_000


def load(path, name, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Source SHA changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prerequisite(h):
    build = h.load_json(MODEL/'build.json')
    parity = h.load_json(MODEL/'parity/result.json')
    parity_completion = h.load_json(MODEL/'parity/completion.json')
    candidate_sha = h.digest(MODEL/'estimator.py')
    h.require(build['passed'] and build['candidate_sha256'] == candidate_sha,
              'Fixed D-only build required')
    h.require(parity['passed'] and parity['completed'] and parity['sources_unchanged']
              and parity_completion['passed']
              and parity_completion['result_sha256'] == h.digest(MODEL/'parity/result.json')
              and parity['candidate_sha256'] == candidate_sha,
              'Exact-source frozen-control numerical parity required')
    h.check_bindings(parity['source_bindings'])
    h.check_bindings(build['source_bindings'])
    h.require(h.digest(BASELINE/'result.json') == BASELINE_SHA, 'Qualified baseline changed')
    baseline = h.load_json(BASELINE/'result.json')
    completion = h.load_json(BASELINE/'completion.json')
    h.require(baseline['passed'] and baseline['hashes_unchanged']
              and baseline['candidate_sha256'] == BASELINE_SOURCE
              and baseline['versions'] == VERSIONS and completion['passed']
              and completion['result_sha256'] == BASELINE_SHA, 'Invalid baseline')
    h.require(h.digest(BASELINE/'predictions_and_bills_sealed.json') == baseline['outputs_seal_sha256'],
              'Baseline output/bill seal changed')
    fixtures,rows = h.fixture_bindings()
    h.require([row['rank'] for row in rows] == [35,100], 'Wrong dev fixtures')
    for row in rows:
        h.require(fixtures['weights_'+str(row['rank'])]['sha256']
                  == baseline['fixture_bindings']['weights_'+str(row['rank'])]['sha256'],
                  'Weight frame differs from baseline')
    bound = {str(path):h.binding(path) for path in
        (MODEL/'build.json',MODEL/'estimator.py',MODEL/'PROTOCOL.md',MODEL/'parity/result.json',
         MODEL/'parity/completion.json',
         HERE/'PROTOCOL.md',Path(__file__).resolve(),HELPER,BASELINE/'result.json',
         BASELINE/'completion.json',BASELINE/'predictions_and_bills_sealed.json')}
    for trial in baseline['trials'][:2]:
        path = BASELINE/trial['output_artifact']
        h.require(h.digest(path) == baseline['artifacts'][path.name]['sha256'], 'Baseline output seal')
        bound[str(path)] = h.binding(path)
    h.check_bindings(bound)
    return candidate_sha,bound,fixtures,rows,baseline


def acquire(h,directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and 'anaconda3' not in sys.executable, 'Use pinned OFFPY, never AP/threadpoolctl')
    h.require(os.getpriority(os.PRIO_PROCESS,0) >= 10, 'nice10 required')
    h.require(all(os.environ.get(k) == '1' for k in BLAS)
              and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1','BLAS1/no bytecode required')
    h.require({k:importlib.metadata.version(k) for k in VERSIONS} == VERSIONS,'Pinned versions')
    candidate_sha,bound,fixtures,rows,baseline = prerequisite(h)
    start = dict(utc=h.utc(),pid=os.getpid(),candidate_sha256=candidate_sha,
                 source_bindings=bound,fixture_bindings=fixtures,versions=VERSIONS,
                 truth_access=False,sequence=[35,100,35],priority=os.getpriority(os.PRIO_PROCESS,0))
    h.write_json(directory/'started.json',start)
    result = dict(passed=False,started=start,candidate_sha256=candidate_sha,trials=[],artifacts={},
                  source_bindings=bound,fixture_bindings=fixtures,truth_access=False,
                  production_residual_validated=False)
    try:
        import numpy as np
        import flopscope as flops
        from threadpoolctl import threadpool_info,threadpool_limits
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = load(MODEL/'estimator.py','fixed_donly_objective_estimator',candidate_sha)
        estimator = None
        def expired(signum,frame):
            raise TimeoutError('Fixed60s per-call development wall cap')
        signal.signal(signal.SIGALRM,expired)
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas'),
                      'Observed BLAS count differs; empty list is not thread-count proof')
            result['threadpool_report'] = pools
            for call,row in enumerate(rows+[rows[0]]):
                weights = np.load(row['weights_path'],mmap_mode='r',allow_pickle=False)
                h.require(weights.shape == (16,1024,1024) and weights.dtype.kind == 'f'
                          and np.isfinite(weights).all(),'Invalid Phase2 weights')
                mlp = MLP(width=1024,depth=16,weights=list(weights),
                          seed=int(derive_estimator_seed(row['mlp_seed'])),name='donly-'+str(row['rank']))
                began = h.utc()
                t0 = time.perf_counter()
                signal.alarm(60)
                try:
                    with flops.BudgetContext(flop_budget=h.BUDGET,wall_time_limit_s=120,quiet=True) as ctx:
                        if estimator is None:
                            estimator = module.Estimator()
                        served = estimator.predict(mlp,h.BUDGET)
                        h.require(isinstance(served,np.ndarray) and served.dtype == np.dtype('float32')
                                  and served.shape == (16,1024),'Concrete servedFP32 contract')
                        materialized = materialise_predictions(served,depth=16,width=1024)
                    elapsed = time.perf_counter()-t0
                finally:
                    signal.alarm(0)
                h.require(served.tobytes() == materialized.tobytes(),'Materialization changed bytes')
                out = np.frombuffer(materialized.tobytes(),dtype=np.float32).reshape(16,1024)
                h.assert_output(np,out)
                old_trial = baseline['trials'][0 if row['rank'] == 35 else 1]
                old_output = np.load(BASELINE/old_trial['output_artifact'],allow_pickle=False)
                h.require(hashlib.sha256(old_output.tobytes()).hexdigest()
                          == old_trial['output_bytes_sha256'],'Baseline served bytes')
                layer0_exact = out[0].tobytes() == old_output[0].tobytes()
                name = f'call_{call:02d}_rank{row["rank"]}_output.npy'
                result['artifacts'][name] = h.save_npy(np,directory/name,out)
                trial = dict(call=call,rank=row['rank'],began_utc=began,ended_utc=h.utc(),
                    flops=int(ctx.flops_used),context_wall_s=elapsed,meter_wall_s=float(ctx.wall_time_s),
                    backend_s=float(ctx.flopscope_backend_time_s),overhead_s=float(ctx.flopscope_overhead_time_s),
                    native_residual_s=float(ctx.residual_wall_time_s),peak_rss_bytes=h.peak_rss_bytes(),
                    output_artifact=name,output_bytes_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                    includes_initialization=call == 0,unchanged_layer0_byte_exact=layer0_exact)
                result['trials'].append(trial)
                bill = f'call_{call:02d}_bill.json'
                h.write_json(directory/bill,trial)
                result['artifacts'][bill] = h.binding(directory/bill)
                print(json.dumps(trial),flush=True)
                h.require(0 < trial['flops'] <= h.BUDGET and elapsed < 60
                          and trial['peak_rss_bytes'] <= RSS_CAP
                          and all(math.isfinite(trial[k]) and trial[k] >= 0 for k in
                                  ('backend_s','overhead_s','native_residual_s')),'Native resource gate')
                h.require(layer0_exact,'Declared unchanged first-Gaussian/P output differs')
                del weights,mlp,served,materialized,out,old_output
        first,last = result['trials'][0],result['trials'][-1]
        h.require(first['output_bytes_sha256'] == last['output_bytes_sha256'],'Repeat output changed')
        h.require(first['flops'] >= last['flops'],'Negative one-time initialization cost')
        result['initialization_flops'] = first['flops']-last['flops']
        for bindings in (bound,fixtures,result['artifacts']):
            h.check_bindings(bindings)
        seal = h.write_json(directory/'predictions_and_bills_sealed.json',dict(
            utc=h.utc(),candidate_sha256=candidate_sha,source_bindings=bound,fixture_bindings=fixtures,
            trials=result['trials'],artifacts=result['artifacts'],truth_access=False,
            repeat_byte_identical=True,initialization_flops=result['initialization_flops']))
        result.update(passed=True,hashes_unchanged=True,outputs_seal_sha256=seal)
    except BaseException:
        result['error'] = traceback.format_exc()
    result['completed_utc'] = h.utc()
    digest = h.write_json(directory/'result.json',result)
    h.write_json(directory/'completion.json',dict(passed=result['passed'],result_sha256=digest))
    print(json.dumps(dict(passed=result['passed'],result_sha256=digest,error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1


def score(h,directory):
    candidate_sha,bound,fixtures,rows,baseline = prerequisite(h)
    acquired = HERE/'native'
    result = h.load_json(acquired/'result.json')
    completion = h.load_json(acquired/'completion.json')
    h.require(result['passed'] and result['hashes_unchanged'] and completion['passed']
              and completion['result_sha256'] == h.digest(acquired/'result.json')
              and result['source_bindings'] == bound and result['fixture_bindings'] == fixtures,
              'Passing same-source acquisition required')
    h.require(result['outputs_seal_sha256'] == h.digest(acquired/'predictions_and_bills_sealed.json'),
              'Candidate output/bill seal')
    h.check_bindings(result['artifacts'])
    import numpy as np
    outputs = {}
    for role,root,trials in (('candidate',acquired,result['trials'][:2]),
                            ('baseline',BASELINE,baseline['trials'][:2])):
        arrays = []
        for trial in trials:
            array = np.load(root/trial['output_artifact'],allow_pickle=False)
            h.assert_output(np,array)
            h.require(hashlib.sha256(array.tobytes()).hexdigest() == trial['output_bytes_sha256'],
                      'Served bytes changed')
            arrays.append(array)
        outputs[role] = arrays
    h.write_json(directory/'outputs_verified_before_truth.json',dict(utc=h.utc(),
        candidate_sha256=candidate_sha,acquisition_sha256=h.digest(acquired/'result.json'),
        baseline_sha256=BASELINE_SHA,truth_access=False))
    truths,truth_bindings = {},{}
    for rank in (35,100):
        path = h.TRUTH_DIR/f'truthpost_rank{rank:04d}.npz'
        truth_bindings[str(rank)] = h.binding(path)
        with np.load(path,allow_pickle=False) as z:
            truth = z['mc_post'].astype(np.float64)
        h.require(truth.shape == (16,1024) and np.isfinite(truth).all(),'Invalid fixed1e9 truth')
        truths[rank] = truth
    roles = {}
    for role,trials in (('candidate',result['trials'][:2]),('baseline',baseline['trials'][:2])):
        per_net = []
        for trial,array in zip(trials,outputs[role],strict=True):
            mse = np.mean((array.astype(np.float64)-truths[trial['rank']])**2,axis=1)
            raw = float(mse[-1]); multiplier = max(.1,trial['flops']/h.BUDGET)
            per_net.append(dict(rank=trial['rank'],raw=raw,layer_mse=mse.tolist(),flops=trial['flops'],
                                multiplier=multiplier,adjusted=raw*multiplier))
        roles[role] = dict(per_net=per_net,mean_raw=sum(t['raw'] for t in per_net)/2,
            mean_flops=sum(t['flops'] for t in per_net)/2,
            mean_adjusted=sum(t['adjusted'] for t in per_net)/2)
    ratio = roles['candidate']['mean_adjusted']/roles['baseline']['mean_adjusted']
    individual = [a['adjusted']/b['adjusted'] for a,b in
                  zip(roles['candidate']['per_net'],roles['baseline']['per_net'],strict=True)]
    advance = bool(ratio < 1 and max(individual) <= 1.1)
    for bindings in (bound,fixtures,result['artifacts'],truth_bindings):
        h.check_bindings(bindings)
    answer = dict(passed=True,hashes_unchanged=True,completed_utc=h.utc(),candidate_sha256=candidate_sha,
        roles=roles,adjusted_ratio=ratio,per_net_adjusted_ratios=individual,
        advances_to_independent14=advance,truth_bindings=truth_bindings,
        reference='Existing1e9 truthpost; fixed development35/100, not a new holdout',
        official_score=False,milestone_achieved=False,production_residual_validated=False,
        aggregation='Mean of own per-net raw*max(.1,F/2^41); initoncecall0, repeat excluded')
    digest = h.write_json(directory/'result.json',answer)
    h.write_json(directory/'completion.json',dict(passed=True,result_sha256=digest))
    print(json.dumps(answer),flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--acquire',action='store_true')
    mode.add_argument('--score',action='store_true')
    args = parser.parse_args()
    if not args.acquire and not args.score:
        parser.print_help(); raise SystemExit(0)
    h = load(HELPER,'donly_fixed_development_utilities',HELPER_SHA)
    directory = HERE/('native' if args.acquire else 'score')
    directory.mkdir(exist_ok=False)
    try:
        raise SystemExit(acquire(h,directory) if args.acquire else score(h,directory))
    except SystemExit:
        raise
    except BaseException:
        h.write_json(directory/'preflight_failure.json',dict(passed=False,utc=h.utc(),error=traceback.format_exc()))
        raise
