"""Inert fixed H15 information test. Never builds a deployable estimator."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
H = HERE/'capture.py'
H_SHA = '95287251980b79d377906c067366714f2a35edd396da00646ebfd37300c98dbe'
CLARIFICATION = HERE/'ANALYSIS_CLARIFICATION.md'
CLARIFICATION_SHA = 'f0f7f79102fdacc2f2ee22b0d895c9e5c789a35c08dee9dec2ac47d379cc9a21'
LABEL_INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
LABEL_INDEX_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
SCHEMAS = dict(unit='current_c2_terminal_quadratic_analysis_unit_v1',
    information='current_c2_terminal_quadratic_information_v1')
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None, truth=False):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2),
            'Regular Phase2-owned file')
    require(truth or ('truthpost' not in path.parts and not
        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower())), 'No implicit truth access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 32*1024**2, 'Bounded regular JSON')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def bind(mapping, tick=None, truth=False):
    for path,digest in mapping.items():
        require(sha(path,tick,truth) == digest, 'Immutable file: '+str(path))


def load(path, digest, name):
    require(sha(path) == digest, 'Exact reviewed source before import')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime(h):
    """Only backend/lifecycle helpers, no candidate or observer imports."""
    bp = h.SHARED/'backend_probe.py'
    backend = load(bp,h.PINS[bp],'h15_analysis_backend_only')
    legacy = load(backend.LEGACY,backend.LEGACY_SHA,'h15_analysis_lifecycle_only')
    return backend,legacy


def positive_chain(folder, inputs, tick=None):
    paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
    ds = [sha(p,tick) for p in paths]
    r,c,f = map(read,paths)
    require(all(v.get('passed') is True and v.get('completed') is True for v in (r,c,f))
        and c['result_sha256'] == f['result_sha256'] == ds[0]
        and f['completion_sha256'] == ds[1], 'Actual positive complete three-seal boundary')
    inputs.update({str(p):d for p,d in zip(paths,ds)})
    return r,ds


def admission(args, tick=None):
    """Stdlib metadata only. In information mode labels are bound, not opened."""
    sources = {str(H):H_SHA,str(CLARIFICATION):CLARIFICATION_SHA,
        str(HERE/'analysis.py'):args.source_sha256,str(HERE/'watch_analysis.py'):args.watcher_sha256,
        str(HERE/'ANALYSIS_PROTOCOL.md'):args.protocol_sha256,
        str(HERE/'ANALYSIS_WATCH_SPEC.md'):args.watch_spec_sha256,
        str(HERE/'analysis_math.py'):args.math_sha256}
    require(args.mode in ('unit','information') and all(re.fullmatch('[0-9a-f]{64}',v or '')
        for v in sources.values()), 'Explicit actual reviewed analysis owners')
    bind(sources,tick)
    h = load(H,H_SHA,'h15_analysis_capture_utilities')
    backend,legacy = runtime(h)
    sources.update({str(p):v for p,v in h.PINS.items()})
    sources.update({str(p):v for p,v in backend._BP_PINS.items()})
    sources.update({str(p):v for p,v in legacy.PINS.items()})
    bind(sources,tick)
    load(HERE/'analysis_math.py',args.math_sha256,'h15_analysis_inert_math')
    inputs,truths = {},{}
    context = dict(mode=args.mode,source_bindings=sources,input_bindings=inputs,
        truth_bindings=truths,candidate_sha256=h.CANDIDATE_SHA)
    if args.mode == 'unit':
        require(not args.capture_manifest and not args.capture_manifest_sha256, 'Target-free unit has no acquisition or label inputs')
    else:
        require(args.capture_manifest and re.fullmatch('[0-9a-f]{64}',args.capture_manifest_sha256 or ''),
                'Actual completed acquisition, not a future guessed hash')
        mp = Path(args.capture_manifest)
        require(mp.resolve().is_relative_to(HERE) and sha(mp,tick) == args.capture_manifest_sha256,
                'Exact all20 H15 capture manifest')
        m = read(mp); inputs[str(mp)] = args.capture_manifest_sha256
        require(m['train'] == list(TRAIN) and len(m['rows']) == 20
            and m['candidate_sha256'] == h.CANDIDATE_SHA and not m['truth_accessed']
            and not m['fit_performed'] and not m['score_measured'], 'All fixed target-free current-c2 TRAIN rows')
        # The manifest explicitly names its positive outer completion. V1's
        # conventional sibling is accepted only if it really exists and passes.
        watch = Path(m.get('watch_folder',str(mp.parent)+'_watch'))
        wr,wd = positive_chain(watch,inputs,tick)
        require(wr['manifest_path'] == str(mp) and wr['manifest_sha256'] == args.capture_manifest_sha256
            and wr['all20_primary_captures_sealed'] and wr['cleanup_passed']
            and wr['whole_owned_tree_retired'] and wr['maximum_swap_growth_bytes'] == 0
            and wr['maximum_interval_s'] <= 1 and wr['maximum_combined_rss_bytes'] < 8_000_000_000,
                'Actual fully retired complete all20 resource boundary')
        for record in (m,wr):
            for group in ('source_bindings','input_bindings','artifacts'):
                bind(record[group],tick)
                inputs.update(record[group])
        for ordinal,row in enumerate(m['rows']):
            require(row['train_ordinal'] == ordinal and row['rank'] == TRAIN[ordinal]
                and row['source_call'] == 0 and row['fresh_cold']
                and 0 < row['own_cold_flops'] <= 2**41
                and row['candidate_sha256'] == h.CANDIDATE_SHA and all(row['head_replay'].values()),
                'Exactly one faithfully captured primary row per TRAIN network')
            for name in ('capture_manifest','capture_npz','model_bill','input_bill','weights'):
                inputs[row[name]] = row[name+'_sha256']
            r,ds = positive_chain(Path(row['result_path']).parent,inputs,tick)
            require(ds == row['worker_seals'] and r['candidate_sha256'] == h.CANDIDATE_SHA,
                    'Every row retains its actual positive worker chain')
            trial,capture = r['trials'][0],r['captures'][0]
            require(trial['call'] == capture['call'] == 0 and trial['rank'] == capture['rank'] == row['rank']
                and trial['flops'] == row['own_cold_flops'] == read(row['model_bill'])['flops']
                and trial['includes_initialization'] is True and trial['wall_s'] < 120
                and row['model_bill'] == str(Path(trial['output']).parent/'call_00_bill.json')
                and row['model_bill_sha256'] == r['artifacts'][row['model_bill']]
                and row['input_bill'] == trial['input_bill'] and row['input_bill_sha256'] == trial['input_bill_sha256']
                and read(row['input_bill'])['weight_file_sha256'] == row['weights_sha256']
                and row['capture_manifest'] == capture['path'] and row['capture_manifest_sha256'] == capture['sha256'],
                    'Row cost/physical weight/capture joins its own actual cold trial, never another network')
            cm = read(row['capture_manifest'])
            require(cm['path'] == row['capture_npz'] and cm['sha256'] == row['capture_npz_sha256']
                and cm['arrays'] == row['arrays'] and cm['head_replay'] == row['head_replay'], 'All eleven arrays join actual captured metadata')
        ur,ud = positive_chain(HERE/'analysis_unit_v1',inputs,tick)
        uw,uwd = positive_chain(HERE/'analysis_unit_watch_v1',inputs,tick)
        verify_result(ur,tick)
        require(ur['mode'] == 'unit' and ur['math_sha256'] == args.math_sha256
            and [uw['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')] == ud
            and uw['schema'] == 'current_c2_terminal_quadratic_analysis_unit_watch_v1'
            and uw['retired_calls'] == 1 and uw['cleanup_passed'] and uw['whole_owned_tree_retired']
            and uw['maximum_combined_rss_bytes'] < 2_000_000_000 and uw['maximum_interval_s'] <= 1
            and uw['maximum_swap_growth_bytes'] == 0 and uw['elapsed_s'] < 120
            and ur['source_bindings'][str(HERE/'analysis_math.py')] == args.math_sha256,
                'Actual same-math target-free unit and complete external boundary')
        require(sha(LABEL_INDEX,tick) == LABEL_INDEX_SHA, 'Original TRAIN target index')
        inputs[str(LABEL_INDEX)] = LABEL_INDEX_SHA
        target_rows = {x['rank']:x for x in read(LABEL_INDEX)['target_sources'] if x['rank'] in TRAIN}
        require(set(target_rows) == set(TRAIN), 'Only the fixed20 original TRAIN labels')
        for row in m['rows']:
            target = target_rows[row['rank']]
            tp = P2/'public_g22_operator_features_v1/truthpost'/f"truthpost_rank{row['rank']:04d}.npz"
            require(target['weight_file'] == row['weights'] and target['target_file'] == str(tp),
                    'Physical capture and original1B target weights join exactly')
            truths[str(tp)] = target['target_sha256']
        context.update(rows=m['rows'],capture_manifest=str(mp),capture_manifest_sha256=args.capture_manifest_sha256,
            acquisition_watch_seals=wd,unit_seals=ud,unit_watch_seals=uwd,target_rows=target_rows)
        bind(inputs,tick)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Inert metadata admission only')
    return context


def unit_math(fnp, math):
    r = fnp.arange(20*1024,dtype=fnp.float64)[:,None]
    c = fnp.arange(64,dtype=fnp.float64)[None,:]
    hidden = fnp.sin(((r+1.0)*(c+1.0))/997.0).astype(fnp.float32).reshape(20,1024,64)
    sigma = fnp.ones((20,1024),dtype=fnp.float64)
    own = fnp.full(20,400_000_000_000,dtype=fnp.int64)
    fit = [i for i in range(20) if i % 5 != 0]
    weight = math.score_weights(fnp,own)
    x,w,frame = math.frame(fnp,hidden,fit,weight)
    design = math.dictionary(fnp,x,sigma,[0])[:16]
    small_w = fnp.full(16,1.0/16,dtype=fnp.float64)
    residual = fnp.sin(fnp.arange(16,dtype=fnp.float64))
    matrix,unused,rms = math.gram(fnp,design,small_w,residual,expected_rows=16)
    checks = {}
    outputs = {}
    for name,quadratic in (('affine',False),('quadratic',True)):
        n = 2145 if quadratic else 65
        known = fnp.sin(fnp.arange(n,dtype=fnp.float64)+1.0)*1e-5
        rhs = (matrix[:n,:n]+fnp.eye(n,dtype=fnp.float64)*1e-4) @ known
        payload,solved = math.solve(fnp,matrix,rhs,rms,quadratic)
        expected_theta = known/rms[:n]
        pieces = [payload['b'].astype(fnp.float64),payload['u'].astype(fnp.float64)]
        if quadratic:
            for i in range(64):
                row = payload['S'][i,i:].astype(fnp.float64)
                mult = fnp.array([1.0]+[2.0]*(63-i),dtype=fnp.float64)
                pieces.append(row*mult)
        observed = fnp.concatenate(pieces)
        maxdiff = fnp.max(fnp.abs(observed-expected_theta),keepdims=True)
        denom = fnp.maximum(fnp.max(fnp.abs(expected_theta),keepdims=True),1e-30)
        error = maxdiff/denom
        require(bool(fnp.all(error <= 2e-6)), 'Known coefficient mapping relative max error <=2e-6')
        raw = fnp.full(1024,0.5,dtype=fnp.float64)
        served = math.served(fnp,x[0],sigma[0],raw,payload)
        # Independent real-monomial FP64 reference for the fixed FP32 circuit.
        reference = math.dictionary(fnp,x,sigma,[0])[:,:n] @ observed
        ideal = fnp.maximum(raw+reference,0.0).astype(fnp.float32)
        circuit_error = fnp.max(fnp.abs(served.astype(fnp.float64)-ideal.astype(fnp.float64)),keepdims=True)
        require(bool(fnp.all(circuit_error <= 1e-6)), 'Dense FP32 versus independent monomial absolute gate')
        checks[name] = dict(solved,known_coefficient_relative_max=float(error[0]),
            dense32_versus_monomial64_maxabs=float(circuit_error[0]))
        outputs[name] = served
    return outputs,dict(checks=checks,synthetic_only=True,actual_training_rows=0,full_model_calls=0,
        parameter_updates=0,truth_accessed=False,information_promising=False)


def decode_inputs(np, fnp, context, truth=True):
    """NP is used only to decode/hash stored bytes; all arithmetic is FNP."""
    collected = {k:[] for k in ('hidden','sigma','raw','served','truth')}
    for row in context['rows']:
        with np.load(row['capture_npz'],allow_pickle=False) as archive:
            arrays = {k:archive[k] for k in archive.files}
        require(set(arrays) == set(row['arrays']), 'Exact eleven-array capture')
        for k,v in arrays.items():
            a = row['arrays'][k]
            require(list(v.shape) == a['shape'] and str(v.dtype) == a['dtype']
                and hashlib.sha256(v.tobytes()).hexdigest() == a['raw_sha256'], 'Captured bytes/type: '+k)
        for k in ('hidden','sigma','raw','served'):
            value = arrays[k][15] if k in ('raw','served') else arrays[k]
            collected[k].append(fnp.array(value))
        if truth:
            path = context['target_rows'][row['rank']]['target_file']
            with np.load(path,allow_pickle=False) as archive:
                value = archive['mc_post']
            require(value.shape == (16,1024) and value.dtype == np.float32, 'Original stored FP32 1B targets')
            collected['truth'].append(fnp.array(value[15]).astype(fnp.float64))
    values = {k:fnp.stack(v) for k,v in collected.items() if v}
    values['own'] = fnp.array([r['own_cold_flops'] for r in context['rows']],dtype=fnp.int64)
    for k,v in values.items():
        math_finite = fnp.all(fnp.isfinite(v))
        require(bool(math_finite), 'Finite fixed information input: '+k)
    baseline = fnp.maximum(values['raw'],0.0).astype(fnp.float32)
    require(baseline.tobytes() == values['served'].tobytes(), 'All20 unchanged comparator replays exact final served bytes')
    return values


def fold_math(np, fnp, math, context, fold):
    data = decode_inputs(np,fnp,context)
    fit = math.fit_fold(fnp,fold,data['hidden'],data['sigma'],data['raw'],data['truth'],data['own'])
    arrays = dict(fit.pop('outputs'))
    arrays.update(fit.pop('normalizer'))
    return arrays,fit


def score_math(np, fnp, context, folder):
    data = decode_inputs(np,fnp,context)
    arrays = {'baseline':data['served']}
    for name in ('affine','quadratic'):
        rows = {}
        for fold in range(5):
            with np.load(folder/f'call_{fold:02d}_outputs.npz',allow_pickle=False) as archive:
                values = archive[name]
            require(values.shape == (4,1024) and values.dtype == np.float32, 'Sealed held dense32 predictions')
            for j,i in enumerate(range(fold,20,5)):
                rows[i] = fnp.array(values[j])
        arrays[name] = fnp.stack([rows[i] for i in range(20)])
    raw_scores,adjusted,fold_scores = {},{},{}
    for name,increment in (('baseline',0),('affine',1_000_000),('quadratic',10_000_000)):
        diff = arrays[name].astype(fnp.float64)-data['truth']
        mse = fnp.mean(diff*diff,axis=1,dtype=fnp.float64)
        mult = fnp.maximum(0.1,(data['own'].astype(fnp.float64)+increment)/(2**41))
        score = mse*mult
        folds = fnp.stack([fnp.mean(score[list(range(k,20,5))],keepdims=True) for k in range(5)]).reshape(5)
        raw_scores[name],adjusted[name],fold_scores[name] = mse,score,folds
    ratio = fnp.mean(adjusted['quadratic'],keepdims=True)/fnp.mean(adjusted['baseline'],keepdims=True)
    beats_affine = fnp.mean(adjusted['quadratic'],keepdims=True) < fnp.mean(adjusted['affine'],keepdims=True)
    wins = (fold_scores['quadratic'] < fold_scores['baseline']) & (fold_scores['quadratic'] < fold_scores['affine'])
    win_count = fnp.sum(wins,keepdims=True)
    promising = bool(fnp.all((ratio <= .99) & beats_affine & (win_count >= 4)))
    output = {}
    summary = {}
    for name in arrays:
        output[name+'_mse'] = raw_scores[name]
        output[name+'_adjusted'] = adjusted[name]
        output[name+'_fold_adjusted'] = fold_scores[name]
        summary[name] = dict(mean_final_mse=float(fnp.mean(raw_scores[name],keepdims=True)[0]),
            mean_adjusted=float(fnp.mean(adjusted[name],keepdims=True)[0]))
    return output,dict(summary=summary,quadratic_to_baseline_adjusted_ratio=float(ratio[0]),
        joint_fold_wins=int(win_count[0]),information_promising=promising,
        exploratory_TRAIN_network_CV_only=True,independent_generalization_evidence=False,
        own_cold_bills=True,actual_paid_inference_bill=False,
        inference_planning_increments=dict(baseline=0,affine=1_000_000,quadratic=10_000_000),
        full_fit_payload_produced=False)


def verify_result(record, tick=None):
    require(record['mode'] in SCHEMAS and record['schema'] == SCHEMAS[record['mode']]
        and all(record[k] for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'all_contexts_retired','all_output_owners_retired','actual_production_backend_qualified'))
        and record['full_model_calls'] == 0 and not record['full_fit_payload_produced']
        and not record['uploaded'] and not record['automatic_next_stage'], 'Completed information only, never release authority')
    expected = ['unit'] if record['mode'] == 'unit' else ['fold0','fold1','fold2','fold3','fold4','score']
    require([v['arm'] for v in record['calls']] == expected and record['retired_call_count'] == len(expected),
            'Every fixed owned context retired once')
    for group in ('source_bindings','input_bindings','artifacts'):
        bind(record[group],tick)
    total = 0
    for call,item in enumerate(record['calls']):
        p = Path(record['output_folder'])/f'call_{call:02d}'
        done = read(str(p)+'_complete.json')
        bill = read(str(p)+'_bill.json'); life = read(str(p)+'_lifecycle.json')
        require(done['passed'] and done['completed'] and done['call'] == call
            and done['arm'] == item['arm'] and done['context_retired']
            and life['context_released'] and life['all_output_owners_retired']
            and life['closed_context_bill_unchanged_by_reset']
            and done['bill_sha256'] == sha(str(p)+'_bill.json',tick)
            and done['lifecycle_sha256'] == sha(str(p)+'_lifecycle.json',tick), 'Own exact closed bill and output lifecycle')
        cap,wall = (12_000_000_000,60) if item['arm'] == 'unit' else ((10_000_000,120) if item['arm'] == 'score' else (350_000_000_000,120))
        require(0 < bill['flops'] <= cap and bill['wall_s'] < wall, 'Original per-context complete cost/time caps')
        bind(done['output_bindings'],tick)
        total += bill['flops']
    require(total == record['research_flops'] <= (12_000_000_000 if record['mode'] == 'unit' else 2_000_000_000_000),
            'All research work counted; not an inference tariff')
    require(record['truth_accessed'] is (record['mode'] == 'information')
        and len(record['truth_bindings']) == (20 if record['mode'] == 'information' else 0), 'Exact label scope')
    if record['mode'] == 'information':
        require(record['truth_inventory_reverified'], 'Worker rehashed original truths after final scoring')
        actual = {x['target_file']:x['target_sha256'] for x in read(LABEL_INDEX)['target_sources'] if x['rank'] in TRAIN}
        require(record['truth_bindings'] == actual, 'Only original fixed20 target metadata; verifier does not open labels')
        seal = read(Path(record['output_folder'])/'all_oof_sealed_before_score.json')
        require(seal['complete_folds'] == 5 and seal['complete_networks'] == 20
            and seal['models'] == ['affine','quadratic'] and not seal['aggregate_score_computed'],
            'All OOF outputs sealed before aggregate scoring')
        bind(seal['output_bindings'],tick)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--mode',choices=('unit','information'))
    parser.add_argument('--capture-manifest')
    for key in ('source','watcher','protocol','watch-spec','math','capture-manifest'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(Path(sys.executable).resolve() == OFFPY.resolve() and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and all(os.environ.get(k) == '1' for k in BLAS) and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and not os.environ.get('PYTHONPATH') and not os.environ.get('FLOPSCOPE_SERVER_URL')
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Original exclusive nice10 single-thread local-only runtime')
    began = time.monotonic()
    cap,rss = (120.,2_000_000_000) if args.mode == 'unit' else (600.,8_000_000_000)
    def tick():
        require(time.monotonic()-began < cap and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < rss,
                'Whole worker time and RSS including publication')
    context = admission(args,tick)
    folder = HERE/('analysis_unit_v1' if args.mode == 'unit' else 'information_v1')
    require(not folder.exists(), 'Exclusive fixed stage; never overwrite or resume')
    folder.mkdir()
    result = dict(schema=SCHEMAS[args.mode],mode=args.mode,passed=False,completed=False,**{k:context[k] for k in
        ('source_bindings','input_bindings','truth_bindings','candidate_sha256')},artifacts={},
        output_folder=str(folder),math_sha256=args.math_sha256,calls=[],retired_call_count=0,
        research_flops=0,truth_accessed=False,full_model_calls=0,parameter_updates=0,full_fit_payload_produced=False,
        uploaded=False,submission_authorized=False,automatic_next_stage=False,information_promising=False)
    if args.mode == 'information':
        result.update(capture_manifest=context['capture_manifest'],capture_manifest_sha256=context['capture_manifest_sha256'])
    try:
        h = load(H,H_SHA,'h15_analysis_native_utilities')
        backend,legacy = runtime(h)
        backend.backend_select()
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_info,threadpool_limits
        flops.configure(symmetry_warnings=False)
        math = load(HERE/'analysis_math.py',args.math_sha256,'h15_fixed_analysis_math')
        with threadpool_limits(limits=1):
            backend.backend_threadpool_info(threadpool_info,result,tick)()
            if args.mode == 'information':
                bind(context['truth_bindings'],tick,truth=True)
                result['truth_accessed'] = True
            def run(call, arm, flop_cap, wall_cap, job):
                start = time.monotonic(); failure = None; packed = {}; metadata = {}; refs = []
                with flops.BudgetContext(flop_budget=flop_cap,wall_time_limit_s=wall_cap,quiet=True) as ctx:
                    try:
                        arrays,metadata = job()
                        for name,value in arrays.items():
                            require(bool(fnp.all(fnp.isfinite(value))), 'Finite final context output')
                            refs.append(weakref.ref(value))
                            packed[name] = (value.tobytes(),str(value.dtype),value.shape)
                        del value,arrays
                    except BaseException:
                        failure = traceback.format_exc()
                p = folder/f'call_{call:02d}'
                bill = dict(legacy.metrics(ctx,start),call=call,arm=arm,flop_cap=flop_cap,
                    research_only=True,charged_to_participant=False)
                bp = Path(str(p)+'_bill.json'); write(bp,bill)
                result['artifacts'][str(bp)] = sha(bp)
                result['research_flops'] += bill['flops']
                ref = weakref.ref(ctx)
                life = legacy.reset_saved_history(flops,ctx,bp)
                del ctx
                gc.collect()
                life.update(complete=failure is None,context_released=ref() is None,
                    all_output_owners_retired=all(v() is None for v in refs))
                lp = Path(str(p)+'_lifecycle.json'); write(lp,life)
                result['artifacts'][str(lp)] = sha(lp)
                require(failure is None, 'Paid context failed: '+str(failure))
                require(life['context_released'] and life['all_output_owners_retired']
                    and life['closed_context_bill_unchanged_by_reset'] and 0 < bill['flops'] <= flop_cap
                    and bill['wall_s'] < wall_cap, 'Actual closed context/array/cost gates')
                op = Path(str(p)+'_outputs.npz')
                decoded = {k:np.frombuffer(b,dtype=d).reshape(s) for k,(b,d,s) in packed.items()}
                with op.open('xb') as stream:
                    np.savez(stream,**decoded)
                del decoded,packed
                dp = Path(str(p)+'_diagnostics.json'); write(dp,metadata)
                outputs = {str(op):sha(op),str(dp):sha(dp)}
                result['artifacts'].update(outputs)
                complete = dict(passed=True,completed=True,call=call,arm=arm,context_retired=True,
                    bill_sha256=sha(bp),lifecycle_sha256=sha(lp),diagnostic_sha256=sha(dp),output_bindings=outputs)
                cp = Path(str(p)+'_complete.json'); write(cp,complete)
                result['artifacts'][str(cp)] = sha(cp)
                result['calls'].append(dict(call=call,arm=arm,flops=bill['flops'],wall_s=bill['wall_s'],complete_sha256=sha(cp)))
                result['retired_call_count'] += 1
                print(json.dumps(dict(event='retired',call=call,arm=arm,complete_sha256=sha(cp))),flush=True)
                tick()
                return metadata
            if args.mode == 'unit':
                result['unit'] = run(0,'unit',12_000_000_000,60,lambda:unit_math(fnp,math))
            else:
                for fold in range(5):
                    run(fold,f'fold{fold}',350_000_000_000,120,lambda:fold_math(np,fnp,math,context,fold))
                before = {str(folder/f'call_{k:02d}_outputs.npz'):sha(folder/f'call_{k:02d}_outputs.npz') for k in range(5)}
                seal = folder/'all_oof_sealed_before_score.json'
                write(seal,dict(complete_folds=5,complete_networks=20,models=['affine','quadratic'],
                    output_bindings=before,aggregate_score_computed=False))
                result['artifacts'][str(seal)] = sha(seal)
                bind(before,tick)
                result['information'] = run(5,'score',10_000_000,120,lambda:score_math(np,fnp,context,folder))
                result['information_promising'] = result['information']['information_promising']
                bind(result['truth_bindings'],tick,truth=True)
                result['truth_inventory_reverified'] = True
        backend.backend_origins(result,tick)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            all_contexts_retired=True,all_output_owners_retired=True,actual_production_backend_qualified=True,
            own_complete_bills=True,all_numerical_array_owners_retired=True,
            all_parameter_and_estimator_owners_retired=True,estimator_instantiated=False)
        verify_result(result,tick)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    gc.collect()
    if time.monotonic()-began >= cap or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= rss:
        result.update(passed=False,completed=False,terminal_resource_check_failed=True)
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(folder/'result.json',result); rh = sha(folder/'result.json')
    ok = bool(result['passed'] and time.monotonic()-began < cap)
    write(folder/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(folder/'completion.json')
    elapsed = time.monotonic()-began; ok = bool(ok and elapsed < cap)
    write(folder/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    print(json.dumps(dict(passed=ok,completed=ok,mode=args.mode,research_flops=result['research_flops'],
        elapsed_s=elapsed,information_promising=result['information_promising'],error=result.get('error'))),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
