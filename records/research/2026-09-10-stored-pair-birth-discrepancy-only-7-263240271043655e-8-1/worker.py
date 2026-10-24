"""One unchanged-COMP35 target-free stored-gauge SIZE call. Inert by default."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
BD = NEW/'birth_direction_fit_v1'
TL = NEW/'current_d50_terminal_linear_v2'
RC = NEW/'current_d50_response10_candidate_v1'
SITE = P2/'data_gate/.venv/lib/python3.11/site-packages'
OFFPY = P2/'data_gate/.venv'
ROW = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0035'
CANDIDATE = COMP/'build_v1/terminal_estimator.py'
CANDIDATE_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
CONTROL = COMP/'panel_v2/terminal'
CONTROL_WATCH = COMP/'panel_v2/terminal_watch'
OUTPUT = HERE/'native_v1'
SCHEMA = 'current_comp_birth_gauge_size_v1'
BASELINE_F, BASELINE_OPS, INITIALIZATION = 471970647528, 31324, 1383914298
DIAGNOSTIC_CAP, TEMP_CAP, VECTOR_CAP = 2044723200, 96*1024**2, 15*32*1024*8
WHOLE_CAP, RSS_CAP = 180., 8_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
PINS = {
    CANDIDATE:CANDIDATE_SHA,
    COMP/'native.py':'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
    COMP/'native_v2.py':'63b2ea4edf5fd67f74a30b7fdee1ab3201a87ca13405654fb1ad0d476f0f3e34',
    RC/'native.py':'cccd8b50d18e6d093a48a3e3675aa20cbe2ef31cb45c2f167f153ddc842fe09c',
    BD/'native_v2.py':'8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    TL/'native.py':'744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe',
    SITE/'flopscope/data/default_weights.json':'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a',
    SITE/'flopscope/_pointwise.py':'7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7',
    SITE/'flopscope/_dtype_billing.py':'d6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8',
    SITE/'flopscope/_accumulation/_reduction.py':'a60b2ef74d568c1a1db5ad0cc06edcdea438efa0fdec7830b17bf439bdf65536',
}
INPUT_PINS = {
    CONTROL/'result.json':'de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625',
    CONTROL/'completion.json':'0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073',
    CONTROL_WATCH/'result.json':'cc3502f464e4b77c15398188515a46ff4e421c3267378382ac0d833bf8bdb0ee',
    CONTROL_WATCH/'completion.json':'2089674c586ea6ccc758c1435574cb57dc5520d55cb5d8ad8645abcc15673fda',
    ROW/'row_manifest.json':'5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    ROW/'weights.npy':'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
}
BEGAN = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def deadline():
    if BEGAN is not None:
        require(time.monotonic()-BEGAN < WHOLE_CAP, 'Whole180s including source admission/seal I/O')


def sha(path):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
        (path.suffix in ('.npy','.npz') and ('truth' in path.name.lower() or path.suffix == '.npz')),
        'Target-free direct source/input/output hashing only')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            deadline(); digest.update(block)
    return digest.hexdigest()


def read(path):
    require(Path(path).suffix == '.json' and Path(path).stat().st_size < 16*1024**2,
            'Bounded metadata only')
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bind(mapping):
    for path, digest in mapping.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}', digest or '')
                and sha(path) == digest, 'Frozen direct owner: '+str(path))


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed inert source owner')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function_source(path, name):
    source = Path(path).read_text()
    nodes = [x for x in ast.parse(source).body
             if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == name]
    require(len(nodes) == 1, 'Unique small source owner: '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def splice(source, edits):
    changed = source
    for before, after in edits:
        require(changed.count(before) == 1, 'Unique forward source span')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique inverse source span')
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Whole small-owner byte inverse')
    return changed


def admission(args):
    sources = {str(path):digest for path,digest in PINS.items()}
    sources.update({str(HERE/'worker.py'):args.source_sha256,
        str(HERE/'gauge_fnp.py'):args.gauge_source_sha256,
        str(HERE/'PROTOCOL.md'):args.protocol_sha256,
        str(HERE/'watch.py'):args.watcher_sha256,
        str(HERE/'WATCH_SPEC.md'):args.watch_spec_sha256})
    inputs = {str(path):digest for path,digest in INPUT_PINS.items()}
    bind(sources); bind(inputs)
    record, end = read(CONTROL/'result.json'), read(CONTROL/'completion.json')
    watch, watch_end = read(CONTROL_WATCH/'result.json'), read(CONTROL_WATCH/'completion.json')
    flags = ('passed','completed','hashes_unchanged','sources_unchanged','own_complete_bills',
             'all16_domains_qualified','all47_selectors_qualified','repeat_output_domain_byte_exact',
             'initialization_ownership_passed')
    require(record['schema'] == 'current_p1_node_composition_independent14_native_v2'
        and all(record.get(k) is True for k in flags) and record['candidate_sha256'] == CANDIDATE_SHA
        and record['truth_accessed'] is False and record['initialization_flops'] == INITIALIZATION
        and len(record['trials']) == len(record['input_preparation']) == 15,
        'Actual qualified current-COMP panel, not a reconstructed ancestor')
    for tail, digest in ((end,INPUT_PINS[CONTROL/'result.json']),
                         (watch_end,INPUT_PINS[CONTROL_WATCH/'result.json'])):
        require(tail['passed'] is True and tail['completed'] is True
                and tail['result_sha256'] == digest, 'Actual historical two-seal chain')
    require(watch['schema'] == 'current_p1_node_composition_independent14_watch_v2'
        and all(watch.get(k) is True for k in ('passed','completed','hashes_unchanged',
                 'sources_unchanged','whole_owned_tree_retired','child_execution_completed'))
        and watch['child_result_sha256'] == INPUT_PINS[CONTROL/'result.json']
        and watch['child_completion_sha256'] == INPUT_PINS[CONTROL/'completion.json']
        and watch['maximum_sampled_rss_bytes'] < RSS_CAP
        and watch['maximum_swap_growth_bytes'] == 0 and watch['maximum_sample_interval_s'] <= 1,
        'Actual completed independent COMP resource supervision')
    trial, prepared = record['trials'][0], record['input_preparation'][0]
    require(trial['call'] == prepared['call'] == 0 and trial['rank'] == prepared['rank'] == 35
        and trial['includes_initialization'] is True and trial['flops'] == BASELINE_F
        and trial['operation_count'] == BASELINE_OPS and prepared['flops'] == 16*1024**2
        and prepared['weight_file_sha256'] == INPUT_PINS[ROW/'weights.npy'], 'Exact coldTRAIN35 control')
    files = ('call_00_input_bill.json','call_00_input_reset.json','call_00_bill.json',
             'call_00_lifecycle.json','call_00_rank35.npy','call_00_diagnostics.json')
    for name in files:
        path = CONTROL/name
        inputs[str(path)] = record['artifacts'][str(path)]
    bind(inputs)
    require(read(CONTROL/'call_00_bill.json') == trial and
            read(CONTROL/'call_00_input_bill.json') == prepared, 'Own archived bill files')
    diagnostic = read(CONTROL/'call_00_diagnostics.json')
    require(diagnostic['startup']['observed_startup_region_flops'] == INITIALIZATION
            and diagnostic['domains']['variance_count'] == 16
            and diagnostic['domains']['selector_count'] == 47, 'Actual source-owned cold startup and all16/47')
    row = read(ROW/'row_manifest.json')
    require(row['role_rank'] == 35 and row['mlp_seed'] == 2758000291903877946
        and row['files']['weights.npy']['sha256'] == INPUT_PINS[ROW/'weights.npy']
        and row['files']['weights.npy']['shape'] == [16,1024,1024]
        and row['files']['weights.npy']['dtype'] == '<f4', 'Exact canonical physical input and seed')
    return dict(source_bindings=sources, input_bindings=inputs, trial=trial,
                prepared=prepared, diagnostic=diagnostic, mlp_seed=row['mlp_seed'])


def prepare():
    """Original paid-loop inverse and exact filtered Observer, no card import."""
    n = load(COMP/'native.py', 'gauge_comp_native_owner', PINS[COMP/'native.py'])
    small = dict(c=SimpleNamespace(require=require), hashlib=hashlib)
    small_text = '\n'.join(function_source(TL/'native.py', name)
                           for name in ('Observer','decode','diagnostics'))
    exec(compile(small_text, str(TL/'native.py')+':exact_small_diagnostics', 'exec'), small)
    n.SMALL = SimpleNamespace(**small)
    # Only prepare is extracted: no endpoint/admission module or old stage runs.
    owner = dict(e=SimpleNamespace(HERE=COMP,function_source=function_source,splice=splice),
                 hashlib=hashlib, __file__=str(COMP/'native_v2.py'))
    text = function_source(COMP/'native_v2.py', 'prepare')
    exec(compile(text, str(COMP/'native_v2.py')+':exact_prepare', 'exec'), owner)
    observer, source, proof = owner['prepare'](n)
    start = source.index("    manifest=json.loads((STORE/'manifest.json').read_text())")
    stop = source.index('    args.out.mkdir()\n', start)
    edits = [
        (source[start:stop], '    rows={35:dict(weights=str(WEIGHT_PATH),mlp_seed=MLP_SEED)}\n'),
        ('            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n',
         '            for call,rank in enumerate(tuple(ranks)):\n'),
        ('                if call==0: first_bytes=values.tobytes()\n'
         "                if call==len(ranks): assert values.tobytes()==first_bytes,'Repeat identity'\n",
         '                # A single cold information call makes no repeat identity claim.\n'),
        ('                del mlp,served,materialized,values\n                gc.collect()\n',
         '                retired_refs=tuple(weakref.ref(x) for x in (mlp,served,materialized,values,*mlp.weights))\n'
         '                del mlp,served,materialized,values\n                gc.collect()\n'
         '                retire_model(result,retired_refs)\n                del retired_refs\n'),
    ]
    source = splice(source, edits)
    proof.update(single_call_roster_inverse_verified=True, no_repeat_claim=True,
        direct_fixture_inverse_verified=True, retirement_strengthened_without_paid_math_change=True,
        single_call_main_sha256=hashlib.sha256(source.encode()).hexdigest(),
        small_diagnostic_source_sha256=hashlib.sha256(small_text.encode()).hexdigest())
    legacy = n.legacy()
    return n, legacy, observer, source, proof


def scalar(item):
    require(tuple(item['shape']) == (), 'Only scalar diagnostic summaries')
    if item['dtype'] == 'bool':
        require(len(item['data']) == 1, 'One boolean scalar')
        return bool(struct.unpack('?',item['data'])[0])
    require(item['dtype'] == 'float64' and len(item['data']) == 8, 'FP64 scalar summary')
    value = struct.unpack('=d',item['data'])[0]
    require(math.isfinite(value), 'Finite paid diagnostic size')
    return value


def gauge_observer(parent, helper):
    class GaugeObserver(parent):
        def __init__(self, module):
            super().__init__(module)
            self.gauge_code = module._bd_update.__code__
            require(self.gauge_code.co_filename == module.__file__, 'Frozen actual BD code owner')

        def begin(self):
            super().begin()
            self.gauge_records = []
            self.diagnostic_flops = self.diagnostic_operations = 0

        def __call__(self, frame, event, answer):
            super().__call__(frame,event,answer)
            if event != 'return' or frame.f_code is not self.gauge_code:
                return
            caller = frame.f_back
            require(caller is not None and caller.f_code is self.codes['_predict_binary'],
                    'Direct actual binary caller, no guessed local namespace')
            local, outer = frame.f_locals, caller.f_locals
            layer = int(outer['l'])
            require(layer == len(self.gauge_records) and 0 <= layer < 15
                    and isinstance(answer,tuple) and len(answer) == 2, 'Exactly15 ordered birth returns')
            values = dict(B=local['bp'], O=local['old'], new32=answer[0], bd_delta=answer[1],
                w=local['weight'], z=local['quadratic_weight'], a32=local['a64'],
                t32=local['t64'], s32=local['s64'], live=outer['live'], alive=local['alive'],
                births=local['births'], born=outer['born'], birth=outer['birth'], beta=outer['beta'],
                u=outer['u'], gap=outer['delta'], vv=outer['vv'], innovation=outer['innovation'],
                safe_birth=outer['safe_birth'], disc=outer['disc'], a=local['a'], t=local['t'], s=local['s'])
            start_f, start_ops = int(self.ctx.flops_used), len(self.ctx.op_log)
            packet = helper.capture(self.module.fnp, values)
            flops = int(self.ctx.flops_used)-start_f
            operations = len(self.ctx.op_log)-start_ops
            require(0 < flops <= 128*1024**2+2048*1024 and operations > 0,
                    'Actual complete per-birth diagnostic tariff within prospective ceiling')
            self.diagnostic_flops += flops; self.diagnostic_operations += operations
            packet.update(layer=layer,flops=flops,operation_count=operations)
            self.gauge_records.append(packet)
            del values, local, outer, caller
    return GaugeObserver


def main():
    global BEGAN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    keys = ('source','gauge-source','protocol','watcher','watch-spec')
    for key in keys:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    # Reject an old/foreign attempt before the failure-publication handler:
    # even false seals must never be appended to an existing stage.
    require(not OUTPUT.exists() and not HERE.is_symlink(), 'Exclusive fresh stage before any publication')
    BEGAN = time.monotonic()
    rc = 1
    try:
        require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '')
                    for k in keys), 'Explicit reviewed source/protocol/watch pins')
        require(Path(sys.prefix).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
            and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Official nice10/BLAS1 native process')
        require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),
                'Admission before all numerical imports')
        require(not OUTPUT.exists() and not HERE.is_symlink(), 'Exclusive one-call output, never resume')
        context = admission(args)
        n, legacy, parent, source, proof = prepare()
        context['source_bindings'].update({str(p):v for p,v in legacy.PINS.items()})
        bind(context['source_bindings'])
        helper = load(HERE/'gauge_fnp.py','gauge_size_public_fnp',args.gauge_source_sha256)
        Observer = gauge_observer(parent,helper)
        versions = {k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
        require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),
                'Unchanged official native package versions')
        require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),
                'All admission/source composition is stdlib-only')

        def after_bill(np, observer, trial, result, folder):
            packet, startup = observer.finish()
            domains = n.SMALL.diagnostics(np,packet)
            domains['selection_cells'] = {key:{k:value[k] for k in
                ('count','floor','indices_sha256','mask_sha256')}
                for key,value in domains['selection_cells'].items()}
            old = context['diagnostic']
            require(domains == old['domains'] and startup == old['startup'],
                    'Exact unchanged all16/all47 and nonoverlapping cold startup')
            require(startup['observed_startup_region_flops'] == INITIALIZATION,
                    'Actual observed startup1,383,914,298, not cold-minus-repeat inference')
            require(trial['output_sha256'] == context['trial']['output_sha256'] and
                    trial['output_bytes_sha256'] == context['trial']['output_bytes_sha256'],
                    'Every actual16-layer served byte equals current COMP coldTRAIN35')
            require(result['input_preparation'][0]['weight_file_sha256'] == INPUT_PINS[ROW/'weights.npy'],
                    'Same exact physical weight bytes')
            records = observer.gauge_records
            require(len(records) == 15 and [r['layer'] for r in records] == list(range(15)), 'All15 birth gauges')
            df, dop = observer.diagnostic_flops, observer.diagnostic_operations
            require(0 < df <= DIAGNOSTIC_CAP and trial['flops']-df == BASELINE_F
                and trial['operation_count']-dop == BASELINE_OPS,
                'Subtract ONLY actual diagnostic counters for unchanged-control verification')
            gauge_path = folder/'gauge_vectors.bin'; offset = 0; saved = []
            with gauge_path.open('xb') as stream:
                for record in records:
                    vectors = record.pop('vectors'); manifest = {}
                    vector_finite = {name:scalar(value) for name,value in record.pop('vector_finite').items()}
                    require(set(vector_finite) == set(helper.VECTOR_NAMES) and all(vector_finite.values()),
                            'Every exported per-hub value, including k/diagonal, is finite')
                    for name in helper.VECTOR_NAMES:
                        raw = vectors.pop(name)
                        stream.write(raw)
                        manifest[name] = dict(offset=offset,bytes=len(raw),dtype='<f8',shape=[1024],
                            raw_sha256=hashlib.sha256(raw).hexdigest())
                        offset += len(raw)
                    stats = {name:{key:scalar(value) for key,value in metric.items()}
                             for name,metric in record.pop('statistics').items()}
                    require(all(metric.get('finite',True) is True for metric in stats.values()),
                            'Every paid matrix finite scan passed')
                    require(record['all_derived_arrays_released'] and record['no_borrowed_array_mutation']
                            and record['explicit_temporary_peak_bytes'] <= TEMP_CAP, 'Bounded nonretaining callback')
                    saved.append(dict(record,statistics=stats,vectors=manifest,all_exported_vectors_finite=True))
            require(offset == VECTOR_CAP, 'Exactly32vectors×15births×1024×8bytes')
            observer.gauge_records.clear()
            stats_path = folder/'gauge_statistics.json'
            write(stats_path,dict(schema=SCHEMA,records=saved,vector_file=str(gauge_path),vector_bytes=offset,
                diagnostic_flops=df,diagnostic_operation_count=dop,truth_accessed=False,
                diagnostic_subtraction_is_control_check_only=True,model_output_unchanged=True))
            dp = folder/'call_00_diagnostics.json'
            write(dp,dict(call=0,rank=35,domains=domains,startup=startup,
                gauge_diagnostic_is_additional_metered_work=True,truth_accessed=False,
                offline_after_saved_bill_and_context_release=True))
            for path in (gauge_path,stats_path,dp):
                result['artifacts'][str(path)] = sha(path)
            result['diagnostic_records'].append(dict(path=str(dp),sha256=sha(dp)))
            result.update(diagnostic_flops=df,diagnostic_operation_count=dop,
                model_flops_after_diagnostic_removal=trial['flops']-df,
                model_operation_count_after_diagnostic_removal=trial['operation_count']-dop,
                baseline_output_byte_exact=True,domains_byte_exact=True,selectors_byte_exact=True,
                all16_domains_qualified=True,all47_selectors_qualified=True,
                initialization_flops=INITIALIZATION,initialization_ownership_passed=True,
                initialization_accounting=dict(observed_startup_region_flops=INITIALIZATION,
                    startup_subtracted_from_bill=False,nested_materializations_not_double_counted=True,
                    cold_minus_repeat_measured=False),
                maximum_explicit_diagnostic_temporary_bytes=max(r['explicit_temporary_peak_bytes'] for r in saved),
                gauge_vector_bytes=offset,all15_gauges_complete=True,all_gauge_arrays_released=True)

        def retire_model(result, refs):
            require(all(ref() is None for ref in refs), 'MLP, input weights, served/materialized arrays retired')
            result.update(full_model_calls=1,retired_call_count=1,all_model_weight_output_owners_retired=True)
            print(json.dumps(dict(event='retired',call=0,rank=35)),flush=True)

        def finish(result):
            require(result['retired_call_count'] == 1 and len(result['trials']) == 1
                and len(result['input_preparation']) == 1 and result['all15_gauges_complete'],
                'One complete retired cold call, never repeat or target acquisition')
            result['artifacts'][str(OUTPUT/'started.json')] = result['started_sha256']
            deadline()
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            require(0 < peak < RSS_CAP, 'Complete native process RSS<8GB')
            result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
                own_complete_bills=True,peak_rss_bytes=peak,elapsed_s=time.monotonic()-BEGAN,
                actual_complete_flops=result['trials'][0]['flops'],
                actual_complete_operation_count=result['trials'][0]['operation_count'])

        header = dict(schema=SCHEMA,stage='size_information',role='unchanged_COMP',completed=False,
            sources_unchanged=False,source_bindings=context['source_bindings'],input_bindings=context['input_bindings'],
            versions=versions,proof=proof,source_inverse_verified=True,
            paid_arithmetic_and_lifecycle_unchanged=True,original_TL_diagnostics_unchanged=True,
            truth_accessed=False,scientific_score_measured=False,score_measured=False,no_fit=True,
            candidate_source_edited=False,parameter_updates=0,counterfactual_evaluated=False,
            diagnostic_subtraction_is_control_check_only=True,full_model_calls=0,retired_call_count=0,
            own_complete_bills=False,repeat_acquired=False,repeat_identity_claimed=False,
            automatic_next_stage=False,submission_authorized=False,raw_benefit_claimed=False)
        namespace = dict(vars(legacy),__file__=str(Path(__file__).resolve()),PARENT=NEW,
            CANDIDATE=CANDIDATE,NATIVE_PINS={**context['source_bindings'],**context['input_bindings']},
            FIXED_RANKS=(35,),WEIGHT_PATH=ROW/'weights.npy',MLP_SEED=context['mlp_seed'],
            Observer=Observer,observed_predict=n.observed_predict,after_bill=after_bill,
            retire_model=retire_model,finish_native=finish,NATIVE_HEADER=header,sha=sha)
        saved_argv = sys.argv
        os.chdir(NEW)
        try:
            sys.argv = [str(__file__),'--source',str(CANDIDATE),'--split','development','--out',str(OUTPUT)]
            exec(compile(source,str(__file__)+':inverse_owned_paid_single_call','exec'),namespace)
            rc = namespace['main']()
        finally:
            sys.argv = saved_argv
    except BaseException:
        error = traceback.format_exc()
        print(error,flush=True)
        if not OUTPUT.exists():
            OUTPUT.mkdir()
        if not (OUTPUT/'result.json').exists():
            write(OUTPUT/'result.json',dict(schema=SCHEMA,passed=False,completed=False,
                error=error,truth_accessed=False,scientific_score_measured=False))
    # False finalization still persists after a caught time failure; no positive
    # seal is possible after180s. This bounded publication has no model activity.
    result_path, completion_path = OUTPUT/'result.json', OUTPUT/'completion.json'
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    record = read(result_path)
    ok = bool(rc == 0 and record.get('passed') and record.get('completed') and time.monotonic()-BEGAN < WHOLE_CAP)
    if not completion_path.exists():
        write(completion_path,dict(passed=ok,completed=ok,result_sha256=digest))
    completion = read(completion_path)
    ok = bool(ok and completion.get('passed') and completion.get('completed')
              and completion['result_sha256'] == digest)
    end_digest = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    elapsed = time.monotonic()-BEGAN
    ok = bool(ok and elapsed < WHOLE_CAP)
    write(OUTPUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=digest,
        completion_sha256=end_digest,elapsed_s=elapsed))
    return 0 if ok and time.monotonic()-BEGAN < WHOLE_CAP else 1


if __name__ == '__main__':
    raise SystemExit(main())
