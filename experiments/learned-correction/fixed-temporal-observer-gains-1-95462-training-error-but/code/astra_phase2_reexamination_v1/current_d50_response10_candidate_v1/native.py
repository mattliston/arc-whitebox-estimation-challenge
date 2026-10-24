"""Fixed response-.1 candidate on35/100/repeat35; inert without --run.

The original official-input/predict/materialize/bill/reset loop is preserved
by an exact inverse. Observation exports existing small bytes and scalar
public counters only. No targets, fit, altered model arithmetic or next stage.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD, TL = NEW/'birth_direction_fit_v1', NEW/'current_d50_terminal_linear_v2'
PROTOCOL_SHA = 'c931293ede873bf193dec3229b7a02583a9ec96a2e2d95996550b9de20861d7f'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
OBSERVER_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
SCHEMA = 'current_d50_response10_candidate_native_v1'
RANKS, BUDGET = (35,100,35), 2**41
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
SMALL = DIAGNOSTICS = None


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No target-array access')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def load(path,name,expected):
    require(sha(path) == expected,'Reviewed source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def legacy():
    return load(BD/'native_v2.py','response10_original_native',NATIVE_SHA)


def authenticate(build_sha256,candidate_sha256,builder_sha256):
    """Replayable, stdlib-only build/source admission; never decode arrays."""
    require(all(re.fullmatch('[0-9a-f]{64}',x or '') for x in
                (build_sha256,candidate_sha256,builder_sha256)), 'Explicit reviewed build/source hashes')
    require(sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA,'Prospective protocol')
    builder = load(HERE/'build.py','response10_builder_admission',builder_sha256)
    context = builder.verify_build(build_sha256,candidate_sha256)
    candidate = context['candidates']['terminal']
    require(Path(candidate['path']) == HERE/'build_v1/estimator.py' and
            candidate['sha256'] == candidate_sha256 and sha(candidate['path']) == candidate_sha256,
            'Sole literal-built response-.1 candidate')
    sources = dict(context['source_bindings'])
    inputs = dict(context['input_bindings'])
    sources.update({str(HERE/'build.py'):builder_sha256,str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,
        str(BD/'native_v2.py'):NATIVE_SHA,str(TL/'native.py'):OBSERVER_SHA,str(TL/'common.py'):COMMON_SHA})
    inputs.update({str(HERE/'build_v1/result.json'):build_sha256,
                   str(HERE/'build_v1/completion.json'):sha(HERE/'build_v1/completion.json')})
    require(all(sha(p) == digest for p,digest in {**sources,**inputs}.items()),'All build/source input seals')
    return dict(context,source_bindings=sources,input_bindings=inputs)


def function_source(path,name):
    text = Path(path).read_text()
    nodes = [x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name == name]
    require(len(nodes) == 1,'Unique frozen function: '+name)
    node = nodes[0]
    return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def load_diagnostics():
    """Only response-selector floor changes in the qualified diagnostics."""
    global SMALL,DIAGNOSTICS
    prior = sys.modules.get('common')
    sys.modules['common'] = load(TL/'common.py','response10_small_common',COMMON_SHA)
    try:
        SMALL = load(TL/'native.py','response10_small_observer',OBSERVER_SHA)
    finally:
        if prior is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common'] = prior
    original = function_source(TL/'native.py','diagnostics')
    before = 'and mask.dtype == np.bool_ and floor == .01'
    after = "and mask.dtype == np.bool_ and floor == (.1 if key.split(':')[1] == '1' else .01)"
    require(original.count(before) == 1,'Unique original selector-floor guard')
    changed = original.replace(before,after)
    require(changed.replace(after,before) == original,'Diagnostic source exact inverse')
    namespace = dict(vars(SMALL))
    exec(compile(changed,str(__file__)+':qualified_diagnostics','exec'),namespace)
    DIAGNOSTICS = namespace['diagnostics']
    return hashlib.sha256(original.encode()).hexdigest()


def bound_depths(module):
    return [] if module._SCRATCH is None else [d for d in range(3,7) if module._SCRATCH[d] is not None]


class Observer:
    """Existing small-byte observer plus source-owned startup counters."""
    def __init__(self,module):
        self.module,self.ctx = module,None
        self.codes = {name:getattr(module,name).__code__ for name in
            ('_predict_learned_compact','_predict_binary','_cp3_bind_arena_views','_bd_coefficients')}
        self.names = {id(getattr(module,name)):name for name in
            ('_PACKED_PUSH','_PACKED_READ','_PACKED_SQUARE','_PACKED_SQUARE4',
             '_PACKED_SQUARE5','_PACKED_SQUARE6')}
        self.begin()

    def begin(self):
        self.ctx = None
        self.small = SMALL.Observer(self.module)
        self.pending,self.bindings,self.coefficients = {},[],[]
        self.prefix = None
        self.depths_before = bound_depths(self.module)

    def __call__(self,frame,event,answer):
        self.small(frame,event,answer)
        code = frame.f_code
        if code is self.codes['_predict_learned_compact'] and event == 'call':
            require(self.prefix is None,'One learned-predict entry')
            self.prefix = dict(start=int(self.ctx.flops_used),P_empty_at_entry=self.module._GNN_PARAMS is None)
        elif code is self.codes['_predict_binary'] and event == 'call':
            require(self.prefix is not None and 'flops' not in self.prefix,'One binary-predict entry')
            self.prefix.update(flops=int(self.ctx.flops_used)-self.prefix.pop('start'),
                               P_ready_at_binary_entry=self.module._GNN_PARAMS is not None)
        elif code is self.codes['_cp3_bind_arena_views']:
            if event == 'call':
                specs = frame.f_locals['view_specs']
                name = self.names.get(id(specs))
                require(name is not None and 'binding' not in self.pending,'Only original owned layout bindings')
                self.pending['binding'] = dict(name=name,start=int(self.ctx.flops_used),
                    expected_flops=sum(stop-start for start,stop,shape in specs),
                    arena_dtype=str(frame.f_locals['arena'].dtype))
            elif event == 'return':
                item = self.pending.pop('binding')
                item['flops'] = int(self.ctx.flops_used)-item.pop('start')
                require(item['arena_dtype'] == 'float32' and item['flops'] == item['expected_flops'],
                        'Original FP32 basic-slice/reshape binding tariff')
                self.bindings.append(item)
        elif code is self.codes['_bd_coefficients']:
            if event == 'call':
                require('coefficient' not in self.pending,'Nonrecursive coefficient materialization')
                self.pending['coefficient'] = dict(start=int(self.ctx.flops_used),
                    empty_at_entry=self.module._BD_COEFFICIENTS is None)
            elif event == 'return':
                item = self.pending.pop('coefficient')
                item['flops'] = int(self.ctx.flops_used)-item.pop('start')
                require(item['flops'] > 0 if item['empty_at_entry'] else item['flops'] == 0,
                        'Original one-time coefficient materialization')
                self.coefficients.append(item)

    def finish(self):
        require(self.ctx is None and not self.pending and self.prefix is not None
                and self.prefix['P_ready_at_binary_entry'] and len(self.coefficients) == 1,
                'Closed/released context and complete startup observations')
        packet = self.small.export_bytes()
        startup = dict(prefix=dict(self.prefix),bindings=list(self.bindings),coefficients=list(self.coefficients),
            square_depths_bound_before=list(self.depths_before),square_depths_bound_after=bound_depths(self.module),
            observed_startup_region_flops=self.prefix['flops']+sum(x['flops'] for x in self.bindings+self.coefficients),
            prefix_scope='Original P materialization plus unchanged zero-state allocation; repeat difference used',
            public_counters_only=True,added_participant_array_operations=0)
        return packet,startup


def observed_predict(estimator,mlp,observer):
    require(sys.getprofile() is None,'No existing profiler may be replaced')
    sys.setprofile(observer)
    try:
        return estimator.predict(mlp,BUDGET)
    finally:
        sys.setprofile(None)


def after_bill(np,observer,trial,result,folder):
    packet,startup = observer.finish()
    diagnostic = DIAGNOSTICS(np,packet)
    diagnostic['selection_cells'] = {key:{k:value[k] for k in
        ('count','floor','indices_sha256','mask_sha256')} for key,value in diagnostic['selection_cells'].items()}
    require(startup['prefix']['P_empty_at_entry'] is (trial['call'] == 0) and
            startup['coefficients'][0]['empty_at_entry'] is (trial['call'] == 0), 'P/cstar initialized once only')
    path = folder/f"call_{trial['call']:02d}_diagnostics.json"
    write(path,dict(call=trial['call'],rank=trial['rank'],domains=diagnostic,startup=startup,
        truth_accessed=False,offline_after_saved_bill_and_context_release=True))
    result['artifacts'][str(path)] = sha(path)
    result['diagnostic_records'].append(dict(path=str(path),sha256=sha(path)))


def finish_native(result):
    require([t['rank'] for t in result['trials']] == list(RANKS) and len(result['diagnostic_records']) == 3,
            'Exactly35/100/repeat35')
    first,middle,last = [read(x['path']) for x in result['diagnostic_records']]
    require(first['domains'] == last['domains'] and
            result['trials'][0]['output_bytes_sha256'] == result['trials'][2]['output_bytes_sha256'],
            'Repeated output/prevariance/ordered-selector bytes exactly equal')
    observed = first['startup']['observed_startup_region_flops']-last['startup']['observed_startup_region_flops']
    measured = result['trials'][0]['flops']-result['trials'][2]['flops']
    require(measured == observed and observed > 0,'Own full cold/repeat difference exactly owned by observed startup')
    require(not last['startup']['bindings'] and not last['startup']['coefficients'][0]['empty_at_entry'],
            'Repeated35 has no new layout/coefficient initialization')
    result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
        initialization_flops=measured,initialization_ownership_passed=True,
        initialization_accounting=dict(cold_minus_repeat_flops=measured,observed_startup_difference_flops=observed,
            per_call_observed_startup_region_flops=[x['startup']['observed_startup_region_flops'] for x in (first,middle,last)],
            startup_subtracted_from_primary_bills=False,old_initialization_value_assumed=False),
        all16_domains_qualified=True,all47_selectors_qualified=True,
        repeat_output_domain_byte_exact=True,own_complete_bills=True)


def derived_main():
    require(sha(BD/'native_v2.py') == NATIVE_SHA,'Frozen paid-loop source')
    original = function_source(BD/'native_v2.py','main')
    start = original.index("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',")
    stop = original.index('    manifest=json.loads(',start)
    edits = [
        (original[start:stop], '    assert candidate == CANDIDATE\n    inputs=dict(NATIVE_PINS)\n'
         '    for path,digest in PINS.items():\n        assert sha(path)==digest,path\n        inputs[str(path)]=digest\n'
         '    ranks=(35,100)\n'),
        ("    write(args.out/'started.json',result)\n", "    result.update(NATIVE_HEADER,diagnostic_records=[])\n"
         "    write(args.out/'started.json',result)\n    result['started_sha256']=sha(args.out/'started.json')\n"),
        ('        estimator=None\n','        observer=Observer(module)\n        estimator=None\n'),
        ('                started=time.monotonic()\n','                observer.begin()\n                started=time.monotonic()\n'),
        ('                    if estimator is None: estimator=module.Estimator()\n',
         '                    observer.ctx=ctx\n                    if estimator is None: estimator=module.Estimator()\n'),
        ('                    served=estimator.predict(mlp,BUDGET)\n',
         '                    served=observed_predict(estimator,mlp,observer)\n'),
        ('                context_ref=weakref.ref(ctx)\n','                observer.ctx=None\n                context_ref=weakref.ref(ctx)\n'),
        ('                if call==0: first_bytes=values.tobytes()\n',
         '                after_bill(np,observer,trial,result,args.out)\n                if call==0: first_bytes=values.tobytes()\n'),
        ('        result.update(passed=True,hashes_unchanged=True,\n'
         "                      initialization_flops=result['trials'][0]['flops']-result['trials'][-1]['flops'])\n",
         '        finish_native(result)\n'),
        ("        result['error']=traceback.format_exc()\n", '        result.update(passed=False,completed=False,'
         'hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())\n'),
        ("    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))\n",
         "    write(args.out/'completion.json',dict(passed=result['passed'],completed=result['completed'],"
         "result_sha256=sha(args.out/'result.json')))\n")]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1,'Unique native source splice')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique native inverse splice')
        inverse = inverse.replace(after,before)
    require(inverse == original,'Every other original input/predict/bill/reset statement unchanged')
    return changed,hashlib.sha256(original.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for name in ('source','builder','build','candidate','protocol'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in
                ('source','builder','build','candidate','protocol')), 'Reviewed digests required')
    require(sha(__file__) == args.source_sha256 and args.protocol_sha256 == PROTOCOL_SHA,'Reviewed worker/protocol')
    require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0) >= 10 and
            all(os.environ.get(k) == '1' for k in BLAS) and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'nice10/BLAS1/assertions/no bytecode or fallback')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'All admission before numerical imports')
    context = authenticate(args.build_sha256,args.candidate_sha256,args.builder_sha256)
    pins = {**context['source_bindings'],**context['input_bindings'],str(Path(__file__).resolve()):args.source_sha256}
    diagnostic_sha = load_diagnostics()
    old = legacy()
    source,base_sha = derived_main()
    require(not (HERE/'native_v1').exists() and not HERE.is_symlink(),'Exclusive native output; no resume')
    namespace = dict(vars(old),__file__=str(Path(__file__).resolve()),PARENT=NEW,
        CANDIDATE=HERE/'build_v1/estimator.py',NATIVE_PINS=pins,Observer=Observer,
        observed_predict=observed_predict,after_bill=after_bill,finish_native=finish_native,
        NATIVE_HEADER=dict(schema=SCHEMA,stage='native',completed=False,sources_unchanged=False,
            build_sha256=args.build_sha256,builder_sha256=args.builder_sha256,truth_accessed=False,no_fit=True,
            native_original_sha256=NATIVE_SHA,native_main_sha256=base_sha,source_inverse_verified=True,
            paid_arithmetic_and_lifecycle_unchanged=True,diagnostics_original_sha256=diagnostic_sha,
            diagnostics_inverse_verified=True,response_floor=.1,active_floor=.01,birth_floor=.01,
            own_complete_bills=False,old_initialization_value_assumed=False,score_measured=False,
            auto_launch_next_stage=False))
    os.chdir(NEW)
    saved = sys.argv
    try:
        sys.argv = [str(__file__),'--source',str(HERE/'build_v1/estimator.py'),
                    '--split','development','--out',str(HERE/'native_v1')]
        exec(compile(source,str(__file__)+':preserved_native','exec'),namespace)
        return namespace['main']()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    raise SystemExit(main())
