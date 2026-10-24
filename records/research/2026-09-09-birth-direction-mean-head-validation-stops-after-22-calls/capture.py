"""Unmodified d50 cold/repeat with passive response-score and read census."""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent,HERE.parent.parent
BD = NEW/'birth_direction_fit_v1'
TL = NEW/'current_d50_terminal_linear_v2'
SOURCE = BD/'build_terminal_v1/estimator.py'
SOURCE_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PROTOCOL_SHA = '57dd4ccedd93c9626a52f0864693a298756bc5e2a06ee2537de99a0ebc4c2dc8'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
CENSUS_SHA = '7e278ed04becda90602773ab607538b0a2330b1e597766a52a391cc6794704cc'
OBSERVER_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
REFERENCE = BD/'panel_v2/terminal/result.json'
REFERENCE_SHA = '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade'
INITIALIZATION = 1383913786
SCHEMA = 'current_d50_response10_census_native_v1'
BUDGET = 2**41
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
SMALL = None


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def load(path,name,expected):
    require(sha(path) == expected,'Frozen source '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def bound_depths(candidate):
    scratch = candidate._SCRATCH
    return [] if scratch is None else [d for d in range(3,7) if scratch[d] is not None]


class Observer:
    """No ndarray arithmetic: byte export, integer shapes and public counters."""
    def __init__(self,candidate):
        self.candidate = candidate
        self.small = SMALL.Observer(candidate)
        self.predict_code = candidate._predict_binary.__code__
        self.select_code = candidate.select.__code__
        self.square_code = candidate._square_product.__code__
        self.ctx = None
        self.tiles,self.reads,self.scores,self.nonresponse_depths = [],{},[],set()
        self.read_active = False
        original_packed,original_mm = candidate._packed_product_cached,candidate._mm
        original_six = candidate._packed_product_6
        def packed(a,b,out,views,levels):
            before = int(self.ctx.flops_used)
            answer = original_packed(a,b,out,views,levels)
            self.tiles.append(dict(shape=[int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])],
                depth=int(levels),dtype=str(a.dtype),flops=int(self.ctx.flops_used)-before))
            return answer
        def six(a,b,out,views):
            before = int(self.ctx.flops_used)
            answer = original_six(a,b,out,views)
            self.tiles.append(dict(shape=[int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])],
                depth=6,dtype=str(a.dtype),flops=int(self.ctx.flops_used)-before))
            return answer
        def mm(a,b):
            caller = sys._getframe(1)
            is_read = caller.f_code is self.predict_code
            layer = int(caller.f_locals['l']) if is_read else None
            del caller
            before = int(self.ctx.flops_used)
            first = len(self.tiles)
            bound_before = bound_depths(candidate)
            shape = [int(a.shape[0]),int(a.shape[1]),int(b.shape[1])]
            if is_read:
                require(not self.read_active and layer not in self.reads,'One direct CP read per layer')
                self.read_active = True
            answer = original_mm(a,b)
            after = int(self.ctx.flops_used)
            # A nonempty native leaf without a packed invocation is direct matmul.
            if max(shape) > 0 and all(shape) and shape[1] <= 8192 and shape[2] <= 8192 and first == len(self.tiles):
                self.tiles.append(dict(shape=shape,depth=0,dtype=str(a.dtype),flops=after-before))
            if is_read:
                self.read_active = False
                self.reads[layer] = dict(shape=shape,read_actual_flops=after-before,
                    read_tiles=[dict(t) for t in self.tiles[first:]],
                    read_square_depths_bound_before=bound_before,
                    read_square_depths_bound_after=bound_depths(candidate))
            return answer
        candidate._packed_product_cached = packed
        candidate._packed_product_6 = six
        candidate._mm = mm

    def begin(self):
        self.ctx = None
        self.small = SMALL.Observer(self.candidate)
        self.tiles.clear(); self.reads.clear(); self.scores.clear(); self.nonresponse_depths.clear()
        self.bound_before = bound_depths(self.candidate)
        self.read_active = False

    def __call__(self,frame,event,answer):
        self.small(frame,event,answer)
        if frame.f_code is self.select_code and event == 'return':
            # Qualified call order: active,response,birth, with no final birth.
            slot = (len(self.small.selectors)-1) % 3
            if slot == 1:
                caller = frame.f_back
                require(caller.f_code is self.predict_code,'Original response selector caller')
                layer = int(caller.f_locals['l'])
                self.scores.append(dict(layer=layer,R=int(caller.f_locals['lam'].shape[0]),
                    a=int(caller.f_locals['live_indices'].shape[0]),q0=int(answer[0].shape[0]),
                    score=SMALL.Observer.bytes_of(frame.f_locals['score']),
                    floor=float(frame.f_locals['floor']),square_depths_bound=bound_depths(self.candidate)))
                del caller
        elif frame.f_code is self.square_code and event == 'return' and not self.read_active:
            depth = int(frame.f_locals['depth'])
            if depth:
                self.nonresponse_depths.add(depth)

    def finish_bytes(self):
        require(not self.read_active and len(self.scores) == 16
                and [s['layer'] for s in self.scores] == list(range(16))
                and set(self.reads) == set(range(1,16)), 'All16 gate rows and15 actual reads')
        return dict(small=self.small.export_bytes(),scores=list(self.scores),reads=dict(self.reads),
            square_depths_bound_before=list(self.bound_before),
            square_depths_bound_after=bound_depths(self.candidate),
            square_depths_used_nonresponse=sorted(self.nonresponse_depths))


def observed_predict(estimator,mlp,observer):
    require(sys.getprofile() is None,'Never replace another profiler')
    sys.setprofile(observer)
    try:
        return estimator.predict(mlp,BUDGET)
    finally:
        sys.setprofile(None)


def after_bill(np,observer,trial,result,folder,reference):
    """Offline small-array decoding only after saved bill/context release."""
    packet = observer.finish_bytes()
    domain = SMALL.diagnostics(np,packet['small'])
    domain['selection_cells'] = {key:{k:item[k] for k in
        ('count','floor','indices_sha256','mask_sha256')} for key,item in domain['selection_cells'].items()}
    require(trial['flops'] == reference['trials'][0]['flops']-(INITIALIZATION if trial['call'] else 0),
            'Actual cold/warm complete d50 bill unchanged')
    require(trial['output_bytes_sha256'] == reference['trials'][0]['output_bytes_sha256'],
            'Every served output byte equals archived current bank')
    arrays,layers = {},[]
    for item in packet['scores']:
        layer,q0,R,a = item['layer'],item['q0'],item['R'],item['a']
        score = SMALL.decode(np,item['score'])
        require(score.dtype == np.float64 and score.shape == (1024,) and np.isfinite(score).all()
                and item['floor'] == .01,'Original exact finite FP64 response score')
        require(q0 == min(1024,((int(np.count_nonzero(score >= .01))+31)//32)*32),
                'Observed original threshold-count rule reconciled')
        q1 = min(1024,((int(np.count_nonzero(score >= .1))+31)//32)*32)
        require(0 <= q1 <= q0 <= a <= 1024,'Fixed stronger response support only')
        arrays[f'response_score_{layer:02d}'] = score
        if R:
            detail = packet['reads'][layer]
            require(detail['shape'] == [q0,R,a], 'Observed read operands match actual gate and bank')
        else:
            require(layer == 0,'Only first layer has no CP columns')
            detail = dict(read_actual_flops=0,read_tiles=[],
                read_square_depths_bound_before=item['square_depths_bound'],
                read_square_depths_bound_after=item['square_depths_bound'])
        layers.append(dict(layer=layer,q0=q0,q1=q1,R=R,a=a,
            score_sha256=hashlib.sha256(item['score']['data']).hexdigest(),
            **{k:v for k,v in detail.items() if k != 'shape'}))
    call = trial['call']
    array_path = folder/f'call_{call:02d}_response_scores.npz'
    with array_path.open('xb') as stream:
        np.savez(stream,**arrays)
    record = dict(schema='current_d50_response10_census_record_v1',rank=35,call=call,
        includes_initialization=trial['includes_initialization'],actual_flops=trial['flops'],
        source_sha256=SOURCE_SHA,response_floor_current=.01,response_floor_proposed=.1,
        layers=layers,domain_diagnostics=domain,score_arrays=str(array_path),score_arrays_sha256=sha(array_path),
        square_depths_bound_before=packet['square_depths_bound_before'],
        square_depths_bound_after=packet['square_depths_bound_after'],
        square_depths_used_nonresponse=packet['square_depths_used_nonresponse'],
        truth_accessed=False,alternate_estimator_executed=False,added_participant_array_operations=0)
    record_path = folder/f'call_{call:02d}_census.json'
    write(record_path,record)
    for path in (array_path,record_path):
        result['artifacts'][str(path)] = sha(path)
    result['census_records'].append(dict(path=str(record_path),sha256=sha(record_path)))
    observer.ctx = None
    return record


def finish_native(result,reference):
    require(len(result['trials']) == len(result['census_records']) == 2,'Two complete calls')
    first,last = [read(x['path']) for x in result['census_records']]
    tree = ast.parse(SOURCE.read_text())
    bindings = {}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name):
            name = node.targets[0].id
            depth_by_name = {'_PACKED_SQUARE': 3, '_PACKED_SQUARE4': 4,
                             '_PACKED_SQUARE5': 5, '_PACKED_SQUARE6': 6}
            if name in depth_by_name:
                bindings[depth_by_name[name]] = sum(stop-start for start,stop,shape in ast.literal_eval(node.value))
    require(set(bindings) == set(range(3,7)),'Source-owned square reshape tariffs')
    for a,b in zip(first['layers'],last['layers']):
        excluded = {'read_actual_flops','read_square_depths_bound_before','read_square_depths_bound_after'}
        require({k:v for k,v in a.items() if k not in excluded} ==
                {k:v for k,v in b.items() if k not in excluded},
                'Exact repeated gate bytes, read geometry and kernel arithmetic')
        setup = lambda row: sum(bindings[d] for d in
            set(row['read_square_depths_bound_after'])-set(row['read_square_depths_bound_before']))
        require(a['read_actual_flops']-setup(a) == b['read_actual_flops']-setup(b),
                'Only actual once-per-depth square bindings may differ in repeated read bill')
    require(first['domain_diagnostics'] == last['domain_diagnostics'],'Exact repeated actual domain')
    init = result['trials'][0]['flops']-result['trials'][1]['flops']
    require(init == INITIALIZATION == reference['initialization_flops'], 'Measured startup ownership')
    result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
        initialization_flops=init,own_output_and_bill_parity=True,all16_domains_qualified=True,
        all47_selectors_qualified=True,repeat_byte_and_bill_exact=True)


def derived_main():
    require(sha(BD/'native_v2.py') == NATIVE_SHA,'Original native loop')
    text = (BD/'native_v2.py').read_text()
    node = next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    start = original.index("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',")
    stop = original.index('    manifest=json.loads(',start)
    edits = [
        (original[start:stop],
         '    assert candidate == CENSUS_SOURCE\n'
         '    inputs=dict(CENSUS_PINS)\n'
         '    for path,digest in PINS.items():\n'
         '        assert sha(path)==digest,path\n'
         '        inputs[str(path)]=digest\n'
         '    ranks=(35,)\n'),
        ("    write(args.out/'started.json',result)\n",
         "    result.update(CENSUS_HEADER,census_records=[])\n"
         "    write(args.out/'started.json',result)\n"
         "    result['started_sha256']=sha(args.out/'started.json')\n"),
        ('        estimator=None\n','        observer=Observer(module)\n        estimator=None\n'),
        ('                started=time.monotonic()\n','                observer.begin()\n                started=time.monotonic()\n'),
        ('                    if estimator is None: estimator=module.Estimator()\n',
         '                    observer.ctx=ctx\n                    if estimator is None: estimator=module.Estimator()\n'),
        ('                    served=estimator.predict(mlp,BUDGET)\n',
         '                    served=observed_predict(estimator,mlp,observer)\n'),
        ('                context_ref=weakref.ref(ctx)\n',
         '                observer.ctx=None\n'
         '                context_ref=weakref.ref(ctx)\n'),
        ('                if call==0: first_bytes=values.tobytes()\n',
         '                after_bill(np,observer,trial,result,args.out,CENSUS_REFERENCE)\n'
         '                if call==0: first_bytes=values.tobytes()\n'),
        ('        result.update(passed=True,hashes_unchanged=True,\n'
         "                      initialization_flops=result['trials'][0]['flops']-result['trials'][-1]['flops'])\n",
         '        finish_native(result,CENSUS_REFERENCE)\n'),
        ("        result['error']=traceback.format_exc()\n",
         '        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,\n'
         '            error=traceback.format_exc())\n'),
        ("    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))\n",
         "    write(args.out/'completion.json',dict(passed=result['passed'],completed=result['completed'],\n"
         "        result_sha256=sha(args.out/'result.json')))\n")]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1,'Unique source-only native splice')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique native inverse splice')
        inverse = inverse.replace(after,before)
    require(inverse == original,'Every remaining native lifecycle/arithmetic statement unchanged')
    return changed,hashlib.sha256(original.encode()).hexdigest()


def main():
    global SMALL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256'); parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(sha(__file__) == args.source_sha256 and args.protocol_sha256 == PROTOCOL_SHA,
            'Reviewed source and prospective protocol')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'Target-free metadata admission')
    require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Fixed nice10/BLAS1/no-fallback frame')
    census = load(NEW/'current_d50_d6_cost_census_v1/run.py','response_census_fixture_owner',CENSUS_SHA)
    pins = {str(p):digest for p,digest in census.FIXED.items()}
    pins.update({str(Path(__file__).resolve()):args.source_sha256,str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,
        str(NEW/'current_d50_d6_cost_census_v1/run.py'):CENSUS_SHA,
        str(TL/'native.py'):OBSERVER_SHA,str(TL/'common.py'):COMMON_SHA})
    require(all(sha(p) == digest for p,digest in pins.items()), 'All frozen sources/weights/fixtures')
    reference = read(REFERENCE)
    require(sha(REFERENCE) == REFERENCE_SHA and reference['passed'] and reference['completed']
            and reference['candidate_sha256'] == SOURCE_SHA and reference['trials'][0]['rank'] == 35
            and reference['trials'][-1]['rank'] == 35,'Completed current-d50 cold/repeat control')
    require(reference['trials'][0]['output_bytes_sha256'] == reference['trials'][-1]['output_bytes_sha256']
            and reference['trials'][0]['flops']-reference['trials'][-1]['flops'] == INITIALIZATION,
            'Actual archived init and all16 repeated output bytes')
    for trial in (reference['trials'][0],reference['trials'][-1]):
        pins[trial['output']] = trial['output_sha256']
    require(all(sha(p) == digest for p,digest in pins.items()), 'Reference outputs and source seals')
    prior = sys.modules.get('common')
    sys.modules['common'] = load(TL/'common.py','response_census_common',COMMON_SHA)
    try:
        SMALL = load(TL/'native.py','response_census_small_observer',OBSERVER_SHA)
    finally:
        if prior is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common'] = prior
    old = load(BD/'native_v2.py','response_census_original_loop',NATIVE_SHA)
    source,base_digest = derived_main()
    require(not (HERE/'native').exists() and not HERE.is_symlink(), 'Exclusive census output, no resume')
    env = dict(vars(old),__file__=str(Path(__file__).resolve()),PARENT=NEW,CENSUS_SOURCE=SOURCE,
        CENSUS_PINS=pins,CENSUS_REFERENCE=reference,Observer=Observer,observed_predict=observed_predict,
        after_bill=after_bill,finish_native=finish_native,
        CENSUS_HEADER=dict(schema=SCHEMA,stage='native',completed=False,sources_unchanged=False,
            alternate_estimator_executed=False,score_measured=False,truth_accessed=False,
            native_original_sha256=NATIVE_SHA,native_main_sha256=base_digest,
            source_inverse_verified=True,paid_arithmetic_and_lifecycle_unchanged=True,
            pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),blas_env=dict.fromkeys(BLAS,'1'),
            response_threshold_proposed=.1,official_score=False,auto_launch_next_stage=False))
    os.chdir(NEW)
    saved = sys.argv
    try:
        sys.argv = [str(__file__),'--source',str(SOURCE),'--split','development','--out',str(HERE/'native')]
        exec(compile(source,str(__file__)+':preserved_native','exec'),env)
        return env['main']()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    raise SystemExit(main())
