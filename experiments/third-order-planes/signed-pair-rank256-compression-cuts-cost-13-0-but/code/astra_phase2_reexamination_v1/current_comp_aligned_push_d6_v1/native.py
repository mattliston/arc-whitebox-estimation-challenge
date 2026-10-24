"""Inert fixed aligned-PUSH CHECK4 acquisition; no new estimator arithmetic."""
import argparse
import ast
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import textwrap
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
CD = NEW/'current_comp_partial_d6_v1'
COMP = NEW/'current_p1_node_composition_v1'
TL = NEW/'current_d50_terminal_linear_v2'
PD = NEW/'current_p1_partial_d6_v1/native.py'
JOINT = NEW/'current_comp_joint_p_node_v1/native.py'
RD = NEW/'current_comp_receiver_directions_v1/native.py'
OFFPY = P2/'data_gate/.venv/bin/python'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
CENSUS_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
KERNEL = NEW/'current_d50_serial_d6_kernel_v1/kernel.py'
KERNEL_SHA = 'a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7'
RANKS = VALID = (37,39,42,48)
FIXED = {'terminal': RANKS}
OUTPUT, ADMISSION = HERE/'native_terminal_v1', HERE/'native_terminal_admission_v1'
SCHEMA = NATIVE_SCHEMA = 'current_comp_aligned_push_d6_native_v1'
CONTEXT_SCHEMA = 'current_comp_aligned_push_d6_native_context_v1'
ADMIT_SCHEMA = 'current_comp_aligned_push_d6_native_admission_v1'
CANDIDATE_SHA = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
CD_SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
BUILD_SHA = 'f841d13473244938a24de14822529960455a2f344306d2899c6656fed4e44f0a'
BUILDER_SHA = 'ef008756a35e3e15e3f0286b6c846ea8d47c63a838d575bb01e54661cbe60f2b'
RUNTIME_SHA = '5c1cb763e99882197d6c0fb6fdf5dc2e7e23b6ae100ac1d16ff60e5cf18b5d9e'
INIT, BUDGET, RSS_LIMIT, EXTRA_BINDINGS = 1944970298, 2**41, 8_000_000_000, 561056000
EXTRA_LAYOUTS = {'_P1_D6_832_PUSH_SPECS':278566656,'_P1_D6_832_READ_SPECS':282489344}
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
OWNERS = {
    CD/'native.py':'f5d0e710ce97e292cd3ecce8f7ebcded3fbdc8194a54fbf8e41d5695e61d483a',
    COMP/'native.py':'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
    PD:'fe5a9d8639b2c2cbd5d1d7886d6c82e953f13e20820161aafb885d5133592e1f',
    JOINT:'851813e9679cd58f1b23bbd15de36cdeb866df0e646e1e8cb9e9aa86e924b0bc',
    RD:'0f107dcd4a6d27da4208a563f0b79c4dcfb966ab8423df3dad492c799edc4a9e',
    NEW/'current_d50_response10_candidate_v1/native.py':'cccd8b50d18e6d093a48a3e3675aa20cbe2ef31cb45c2f167f153ddc842fe09c',
    NEW/'birth_direction_fit_v1/native_v2.py':'8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    TL/'native.py':'744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe',
    TL/'common.py':'52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a',
    TL/'watch_fit.py':'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd',
    CENSUS:CENSUS_SHA, KERNEL:KERNEL_SHA,
    HERE/'build.py':BUILDER_SHA,
    HERE/'DECISION.md':'ff026a7659875606a06206a89f047331a77d86af7da030266be74130bbf784f0',
    HERE/'watch.py':'d6eaa5714b42836b530ae3db153dbef12fb2764f01d8403f917848b0d867bd91',
    HERE/'client.py':'d5335fb2ce26b7ecf4756dad7470ae87bd14911607965feef40228f8cd2f8088',
    HERE/'runner.py':'1f4f47be0265582733862809b5c985c194b263a8032d079ec22eb57fef70c0ce',
    HERE/'ADAPTERS.md':'eaba36a24addff40abb5babe97d5eb25442a87021a23a6dbee43b790d15cf979',
    CD/'build_v1/estimator.py':CD_SHA,
}
CHAINS = {
    HERE/'build_v1':(BUILD_SHA,'e3d2ac39f6a2b0b489eb9a83beab987d2b36623e561d0611fe523fcdaacb0707',
        '9bac7e303cad3ec1a83045e16f18fc6874c40b6c2444372e1d890eb898502ffe'),
    HERE/'runtime_v1':(RUNTIME_SHA,'547cd2a9dc0a84ec7fff630886174ac736efa432e92c1f6009b351757c4cd724',
        '5cfbb63193d8f7e0da7cca38506a0449d92bfad4c4553b5443757dfe65679447'),
    CD/'native40_terminal_v1':('b9434921a668538f2cf434a6df773bca7e42fab94c9acc5f85f1213592f6fc2b',
        '4397428b9cbbe76d1f96a64897ca6ae1dcd65faaf442f8a451ac74b86dd02f05',
        'a93c6670e67bd96ae0f19edf946288adb1ef210d45eb31347b9a51e419febbea'),
    CD/'native40_terminal_watch_v1':('29a6df3cd06b7c0c5e315315c1a85a240dfe0df592ebb26df9408de9fca42e87',
        '3f9f2bf54f585674c3bac14eb99b5561634a9c28f80bcfa6d707a6aa28d46f8a',
        'dcf7a4ea46735f0ec1aacf34a107fc740f6f58dbd291b512bc10ce93d4566493'),
}
_OWNER = _BASE = _FUNCTIONS = None


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and
        not (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No target-array access')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick is not None: tick()
    return digest.hexdigest()


def read(path, bound=4_000_000):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < bound, 'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def bind(items, tick=None):
    require(isinstance(items,dict), 'Flat hash inventory')
    for path,digest in items.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or '')
            and sha(path,tick) == digest, 'Immutable direct dependency: '+str(path))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load(path, name, digest):
    require(sha(path) == digest, 'Frozen source owner')
    return module(path,name)


def owner():
    global _OWNER
    if _OWNER is None: _OWNER = load(CD/'native.py','apush_cd_native_source_owner',OWNERS[CD/'native.py'])
    return _OWNER


def base():
    global _BASE
    if _BASE is None: _BASE = owner().base()
    return _BASE


def function_source(source, name):
    found = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == name]
    require(len(found) == 1, 'Unique small metadata function '+name)
    node = found[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def splice(source, edits):
    return owner().splice(source,edits)


def environment():
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
        and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Official nice10/BLAS1/assertions/no fallback')
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Admission before numerical imports')
    versions = {k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
    require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),
        'Same original four official package versions, checked without numerical imports')


def reviewed_owners(args):
    return {str(p):v for p,v in {**OWNERS,Path(__file__).resolve():args.source_sha256,
        HERE/'NATIVE_PROTOCOL.md':args.protocol_sha256,HERE/'watch_native.py':args.watcher_sha256,
        HERE/'NATIVE_WATCH_SPEC.md':args.watch_spec_sha256}.items()}


def eligible(m,k,n):
    return all(type(v) is int for v in (m,k,n)) and m == 1024 and 0 < k <= 1024 and 1024 < n <= 8192 and k % 64 == n % 64 == 0


def product_flops(m,k,n,depth=6):
    """Exact frozen kernel.py price, including all stack/copy arithmetic."""
    require(all(type(v) is int and v > 0 for v in (m,k,n,depth))
        and all(v % 2**depth == 0 for v in (m,k,n)), 'Exact aligned tariff domain')
    q = Fraction(7,4)
    value = 7**depth*(m//2**depth)*(n//2**depth)*(2*k//2**depth-1)
    value += q*(m*k+k*n+m*n)*sum(q**j for j in range(depth))
    require(value.denominator == 1, 'Integral public FP32 tariff')
    return value.numerator


def validate_dispatches(domain, dispatches):
    expected = []
    cells = domain['selection_cells']
    for layer in range(1,16):
        k = cells[f'{layer-1}:0']['count']
        columns = 2*sum(cells[f'{j}:2']['count'] for j in range(layer-1))
        for start in range(0,columns,8192):
            width = min(8192,columns-start)
            if eligible(1024,k,width): expected.append((layer,[1024,k,width]))
    require([(d['layer'],d['shape']) for d in dispatches] == expected,
        'Exact ordered current-trajectory broad-PUSH census including eligible tails')
    for item in dispatches:
        m,k,n = item['shape']
        require(eligible(m,k,n) and item['dtype'] == 'float32'
            and type(item['flops']) is int and item['flops'] == product_flops(m,k,n)
            and item['same_original_arena'] and item['returned_original_out'] and item['child_bypasses_dispatcher']
            and [x['child'] for x in item['children']] == list(range(7)), 'Complete same-arena serial tariff and owners')
        child_shape = [m//2,k//2,n//2]
        require(all(x['depth'] == 5 and x['shape'] == child_shape
            and type(x['flops']) is int and x['flops'] == product_flops(*child_shape,5)
            for x in item['children']), 'All seven directly owned D5 shapes and complete tariffs')


def observer_sources():
    (domain,comp,original), prior = owner().observer_sources()
    changed = splice(original, [
        ("and (tuple(a.shape),tuple(b.shape),tuple(out.shape)) == ((1024,832),(832,8192),(1024,8192))",
         "and eligible(int(a.shape[0]),int(a.shape[1]),int(b.shape[1]))\n"
         "                    and tuple(b.shape) == (int(a.shape[1]),int(b.shape[1]))\n"
         "                    and tuple(out.shape) == (1024,int(b.shape[1]))"),
        ("'Only fixed existing push dispatch'", "'Only the fixed broad existing PUSH-owner policy'"),
        ("shape=list(SHAPE)", "shape=[int(a.shape[0]),int(a.shape[1]),int(b.shape[1])]"),
        ("item['flops'] == PRODUCT_FLOPS", "item['flops'] == product_flops(*item['shape'])"),
        ("and tuple(frame.f_locals['a'].shape) == (512,416)",
         "and tuple(frame.f_locals['a'].shape) == tuple(v//2 for v in self.serial['shape'][:2])"),
        ("and tuple(frame.f_locals['b'].shape) == (416,4096)",
         "and tuple(frame.f_locals['b'].shape) == tuple(v//2 for v in self.serial['shape'][1:])"),
        ("dict(child=j,start=int(self.ctx.flops_used),depth=5)",
         "dict(child=j,start=int(self.ctx.flops_used),depth=5,shape=[v//2 for v in self.serial['shape']])"),
        ("item['flops'] > 0 and answer is frame.f_locals['out']",
         "item['flops'] == product_flops(*item['shape'],5) and answer is frame.f_locals['out']"),
    ])
    return (domain,comp,changed), dict(parent=prior,extra_observer_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        extra_observer_derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        broad_push_observer_inverse_verified=True,unchanged_all16_all47_capture=True,
        no_added_participant_array_operations=True,exact_shape_specific_D6_and_seven_D5_tariffs=True)


def metadata_sources():
    original, prior = owner().metadata_sources()
    validate = function_source(original,'validate_diagnostic')
    tail = validate[validate.index('    expected = {'):validate.index('    return domain,startup,dispatches')]
    changed = splice(original, [
        (tail, '    validate_dispatches(domain,dispatches)\n'),
        ("measured > 0 and not records[-1]['startup']['bindings']",
         "measured == INIT and not records[-1]['startup']['bindings']"),
        ("        full_model_calls=5,retired_call_count=5,COMP_parameter_initialization_owned=True,",
         "        full_model_calls=5,retired_call_count=5,COMP_parameter_initialization_owned=True,\n"
         "        all_eligible_push_dispatches_qualified=True,runtime_prerequisite_qualified=True,"),
        ("record['build_watch_sha256'] == context['build_watch_sha256']",
         "record['runtime_sha256'] == context['runtime_sha256']"),
        ("and record['COMP_parameter_initialization_owned'] and record['proof'] == context['proof']",
         "and record['COMP_parameter_initialization_owned'] and record['proof'] == context['proof']\n"
         "            and record['all_eligible_push_dispatches_qualified'] and record['runtime_prerequisite_qualified']\n"
         "            and record['runtime_sha256'] == RUNTIME_SHA and record['initialization_flops'] == INIT"),
    ])
    return changed, dict(parent=prior,original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),full_metadata_source_inverse_verified=True)


def callbacks():
    global _FUNCTIONS
    if _FUNCTIONS is None:
        source,_ = metadata_sources(); env = dict(globals())
        exec(compile(source,str(__file__)+':CD_metadata_inverse','exec'),env)
        _FUNCTIONS = env
    return _FUNCTIONS


def after_bill(np, observer, trial, result, folder, refs):
    return callbacks()['after_bill'](np,observer,trial,result,folder,refs)


def verify_acquisition(record, end, digest, args, context, tick=None):
    return callbacks()['verify_acquisition'](record,end,digest,args,context,tick)


def verify_native(record, end, digest, args, context, tick=None):
    verify_acquisition(record,end,digest,args,context,tick)
    final = read(OUTPUT/'finalization.json')
    require(final['passed'] is True and final['completed'] is True
        and final['result_sha256'] == digest and final['completion_sha256'] == sha(OUTPUT/'completion.json',tick)
        and 0 <= final['elapsed_s'] < 600, 'Actual worker third seal and complete time bound')
    return final


def worker_source():
    resolved,_ = owner().lifecycle_source()
    original = function_source(resolved,'run')
    changed = splice(original, [
        ("context = read(ad/'context.json', 4_000_000);", "context = read_context(args,args.context_sha256);"),
        ("HERE/'watch_native.py': args.watcher_sha256}.items()",
         "HERE/'watch_native.py': args.watcher_sha256,\n"
         "        HERE/'NATIVE_WATCH_SPEC.md': args.watch_spec_sha256}.items()"),
        ("build_watch_sha256=context['build_watch_sha256']", "runtime_sha256=context['runtime_sha256']"),
    ])
    return changed, dict(owner_sha256=OWNERS[CD/'native.py'],original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),outer_worker_source_inverse_verified=True)


def prepare(n=None):
    global _BASE
    n = base() if n is None else n; _BASE = n
    if n.SMALL is None: n.load_diagnostics()
    _,_,paid,paid_proof = owner().prepare(n)
    (domain,comp,extra),observer_proof = observer_sources()
    env = dict(vars(n.SMALL))
    exec(compile(domain,str(__file__)+':unchanged_domain_capture','exec'),env)
    env = dict(vars(n),Domain=env['DomainObserver'],EXTRA_LAYOUTS=EXTRA_LAYOUTS,
        EXTRA_BINDINGS=EXTRA_BINDINGS,eligible=eligible,product_flops=product_flops)
    exec(compile(comp+'\n'+extra,str(__file__)+':broad_push_observer','exec'),env)
    _,metadata_proof = metadata_sources(); _,run_proof = worker_source()
    proof = dict(parent_CD=paid_proof,observer=observer_proof,metadata=metadata_proof,outer_worker=run_proof,
        original_paid_input_predict_save_reset_release_byte_exact=True,
        original_all16_all47_diagnostics_unchanged=True,no_added_participant_array_operations=True,
        no_estimator_source_or_parameter_changes=True,kernel_tariff_owner_sha256=KERNEL_SHA)
    return env['Observer'],callbacks()['finish_native'],paid,proof


def seal3(folder, digests, inputs, *, completed=True, final_result_link=True, tick=None):
    paths = [folder/name for name in ('result.json','completion.json','finalization.json')]
    for path,digest in zip(paths,digests):
        require(sha(path,tick) == digest, 'Exact previously completed seal')
        inputs[str(path)] = digest
    record,end,final = [read(p) for p in paths]
    require(all(x.get('passed') is True for x in (record,end,final))
        and (not completed or all(x.get('completed') is True for x in (record,end,final)))
        and end['result_sha256'] == digests[0] and final['completion_sha256'] == digests[1]
        and (not final_result_link or final['result_sha256'] == digests[0]), 'Actual historical seal schema/links')
    return record,end,final


def runtime_admission(args, sources, inputs, tick):
    require((args.runtime_sha256,args.build_sha256,args.candidate_sha256) == (RUNTIME_SHA,BUILD_SHA,CANDIDATE_SHA),
        'Only the completed fixed-policy runtime/build/candidate')
    runtime,_,rf = seal3(HERE/'runtime_v1',CHAINS[HERE/'runtime_v1'],inputs,tick=tick)
    built,_,_ = seal3(HERE/'build_v1',CHAINS[HERE/'build_v1'],inputs,tick=tick)
    flags = ('passed','completed','runtime_falsifier_passed','repeat_byte_exact','hashes_unchanged','sources_unchanged',
        'whole_owned_tree_retired','child_execution_completed','cleanup_passed','observational_groups_retired',
        'actual_completed_samples','foreign_numerical_workers_checked')
    require(runtime['schema'] == 'current_comp_aligned_push_d6_runtime_v1' and all(runtime.get(k) is True for k in flags)
        and runtime['candidate_sha256'] == CANDIDATE_SHA and runtime['ranks'] == [35,35]
        and runtime['full_model_calls'] == 2 and runtime['initialization_flops'] == INIT
        and 0 < runtime['elapsed_s'] <= rf['elapsed_s'] < 300
        and 0 < runtime['maximum_combined_rss_bytes'] < RSS_LIMIT and 0 < runtime['maximum_interval_s'] <= 1
        and runtime['maximum_swap_growth_bytes'] == 0 and 0 <= rf['terminal_publication_interval_s'] <= 1
        and all(runtime[k] is False for k in ('numerical_imports','truth_arrays_accessed','raw_error_measured',
            'adjusted_score_measured','production_five_qualified','uploaded','automatic_next_stage')),
        'Completed limited target-free runtime falsifier, not raw evidence')
    require(runtime['build_seals'] == dict(zip(('result_sha256','completion_sha256','finalization_sha256'),
        CHAINS[HERE/'build_v1'])) and built['schema'] == 'current_comp_aligned_push_d6_build_v1'
        and built['candidate_sha256'] == CANDIDATE_SHA and built['estimator_imported'] is False
        and built['truth_accessed'] is False and built['score_measured'] is False, 'Build enclosed by successful whole-stage supervisor')
    proof = built['proof']
    require(proof['parent_sha256'] == CD_SHA and proof['candidate_sha256'] == CANDIDATE_SHA
        and proof['existing_complete_initialization_flops'] == INIT and proof['additional_initialization_over_CD'] == 0
        and proof['backing_arena_bytes'] == 1168900096 and all(proof[k] is True for k in
            ('whole_source_inverse_verified','only_final_dispatch_predicate_changed','heads_and_physics_byte_unchanged',
             'same_original_backing_arena','no_read_or_square_change','no_padding','no_rank_or_policy_tuning')),
        'Previously sealed unique predicate-only source inverse')
    candidate = HERE/'build_v1/estimator.py'
    require(built['artifacts'] == {str(candidate):CANDIDATE_SHA}
        and runtime['artifacts'][str(candidate)] == CANDIDATE_SHA, 'Same actual built source in both prior receipts')
    for group,target in (('source_bindings',sources),('input_bindings',inputs),('artifacts',inputs)):
        bind(runtime[group],tick); target.update(runtime[group])
    sources[str(candidate)] = CANDIDATE_SHA
    require(all(sources.get(str(p)) == h for p,h in OWNERS.items() if p.parent == HERE and p.name != 'native.py'),
        'Every frozen APUSH implementation owner agrees with runtime')
    return proof


def references(sources, inputs, tick):
    folder = CD/'native40_terminal_v1'
    record,end,_ = seal3(folder,CHAINS[folder],inputs,tick=tick)
    wf = CD/'native40_terminal_watch_v1'
    watch,_,final = seal3(wf,CHAINS[wf],inputs,completed=False,final_result_link=False,tick=tick)
    require(record['schema'] == 'current_comp_partial_d6_native40_v1' and record['candidate_sha256'] == CD_SHA
        and record['truth_accessed'] is False and record['full_model_calls'] == record['retired_call_count'] == 41
        and len(record['trials']) == len(record['input_preparation']) == len(record['diagnostic_records']) == 41
        and all(record[k] is True for k in ('own_complete_bills','all16_domains_qualified','all47_selectors_qualified',
            'initialization_ownership_passed','same_existing_arena_qualified','dispatch_tariffs_qualified'))
        and record['initialization_flops'] == INIT, 'Qualified same-parent native40 reference boundary')
    require(watch['schema'] == 'current_comp_partial_d6_native40_watch_v1' and watch['retired_calls'] == 41
        and watch['whole_owned_tree_retired'] and watch['cleanup_passed'] and watch['sources_unchanged']
        and watch['numerical_imports'] is False and watch['elapsed_s'] < 1900
        and 0 < watch['maximum_combined_rss_bytes'] < RSS_LIMIT and 0 < watch['maximum_interval_s'] <= 1
        and final['elapsed_through_completion_io_s'] < 1900
        and tuple(watch[k] for k in ('child_result_sha256','child_completion_sha256','child_finalization_sha256')) == CHAINS[folder],
        'Actual historical41 supervisor schema, no invented completed/swap/final fields')
    old = base().legacy(); sources.update({str(p):h for p,h in old.PINS.items()})
    helper = load(CENSUS,'apush_old_lifecycle_receipts',CENSUS_SHA)
    store = old.STORE; manifest = store/'manifest.json'
    inputs[str(manifest)] = base().STORE_SHA
    require(sha(manifest,tick) == base().STORE_SHA, 'Original canonical input manifest')
    rows = read(manifest)['roles']['observer_fit']['row_manifests']; refs = {}
    for call,rank in enumerate(RANKS):
        trial,prepared,diag = record['trials'][call],record['input_preparation'][call],record['diagnostic_records'][call]
        require(trial['call'] == prepared['call'] == call and trial['rank'] == prepared['rank'] == rank
            and trial['includes_initialization'] is (call == 0), 'Exact same first-four cold history')
        paths = (folder/f'call_{call:02d}_bill.json',folder/f'call_{call:02d}_input_bill.json',
            folder/f'call_{call:02d}_lifecycle.json',folder/f'call_{call:02d}_input_reset.json',Path(trial['output']),Path(diag['path']))
        require(read(paths[0]) == trial and read(paths[1]) == prepared
            and paths[4] == folder/f'call_{call:02d}_rank{rank}.npy'
            and paths[5] == folder/f'call_{call:02d}_diagnostics.json', 'Canonical direct comparator artifacts')
        for path in paths:
            digest = record['artifacts'][str(path)]
            require(sha(path,tick) == digest, 'Selected reference file inside its qualified result')
            inputs[str(path)] = digest
        require(trial['output_sha256'] == inputs[str(paths[4])] and diag['sha256'] == inputs[str(paths[5])],
            'Reference output/diagnostic exact seals')
        base().npy_bytes(paths[4],trial['output_bytes_sha256'])
        owner().callbacks()['validate_diagnostic'](read(paths[5]),call)
        helper.lifecycle(dict(trial,lifecycle_receipt=str(paths[2])),paths[0],paths[2])
        helper.lifecycle(dict(prepared,lifecycle_receipt=str(paths[3])),paths[1],paths[3])
        matches = [row for row in rows if row['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical physical input')
        item = matches[0]; rp = store/item['path']; row = read(rp); wp = rp.parent/'weights.npy'
        require(sha(rp,tick) == item['sha256'] and row['files']['weights.npy']['sha256'] == prepared['weight_file_sha256'],
            'Same canonical stored FP32 weights, not another panel input')
        inputs.update({str(rp):item['sha256'],str(wp):prepared['weight_file_sha256']})
        refs[rank] = dict(trial,trial=trial,weight_file_sha256=prepared['weight_file_sha256'],
            input_preparation=prepared,diagnostic=diag,diagnostics=diag)
    return refs


def authenticate(args, tick):
    sources = reviewed_owners(args); inputs = {}
    bind(sources,tick)
    build_proof = runtime_admission(args,sources,inputs,tick)
    refs = references(sources,inputs,tick)
    _,_,_,proof = prepare()
    bind(sources,tick); bind(inputs,tick); environment(); tick()
    return dict(schema=CONTEXT_SCHEMA,role='terminal',ranks=list(RANKS),
        candidate=str(HERE/'build_v1/estimator.py'),candidate_sha256=CANDIDATE_SHA,build_sha256=BUILD_SHA,
        builder_sha256=BUILDER_SHA,runtime_sha256=RUNTIME_SHA,source_bindings=sources,input_bindings=inputs,
        reference_by_rank=refs,proof=proof,build_proof=build_proof,admission_pid=os.getpid(),
        source_admission_complete=True,runtime_prerequisite_qualified=True,truth_accessed=False,numerical_imports=False)


def admit(args):
    environment(); began = time.monotonic()
    require(not ADMISSION.exists() and not HERE.is_symlink(), 'Exclusive retired admission, no retry')
    ADMISSION.mkdir()
    result = dict(schema=ADMIT_SCHEMA,passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
        role='terminal',admission_pid=os.getpid(),numerical_imports=False,truth_accessed=False,
        source_bindings=reviewed_owners(args),input_bindings={},artifacts={})
    def guard(*unused):
        require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
            'Source-only admission60s/2GB')
    handler = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        context = authenticate(args,guard)
        require(len(json.dumps(context,allow_nan=False).encode()) < 4_000_000, 'Bounded metadata-only context')
        write(ADMISSION/'context.json',context); digest = sha(ADMISSION/'context.json',guard)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            context_sha256=digest,source_bindings=context['source_bindings'],input_bindings=context['input_bindings'],
            artifacts={str(ADMISSION/'context.json'):digest},runtime_prerequisite_qualified=True,proof=context['proof'])
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,handler)
        result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if result['elapsed_s'] >= 60 or result['peak_rss_bytes'] >= 2_000_000_000:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False)
        write(ADMISSION/'result.json',result); rh = sha(ADMISSION/'result.json')
        ok = bool(result['passed'] and time.monotonic()-began < 60)
        write(ADMISSION/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(ADMISSION/'completion.json')
        elapsed = time.monotonic()-began; ok = bool(ok and elapsed < 60)
        write(ADMISSION/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
        sha(ADMISSION/'finalization.json'); ok = bool(ok and time.monotonic()-began < 60)
    print(json.dumps(dict(passed=ok,context_sha256=result.get('context_sha256'),error=result.get('error'))),flush=True)
    return 0 if ok else 1


def read_context(args, digest, tick=None):
    require(args.role == 'terminal' and digest == args.context_sha256
        and sha(ADMISSION/'context.json',tick) == digest, 'Explicit fixed retired context')
    context = read(ADMISSION/'context.json')
    result,end,final = [read(ADMISSION/name) for name in ('result.json','completion.json','finalization.json')]
    require(context['schema'] == CONTEXT_SCHEMA and result['schema'] == ADMIT_SCHEMA
        and all(x.get(k) is True for x in (result,end,final) for k in ('passed','completed'))
        and result['hashes_unchanged'] and result['sources_unchanged']
        and end['result_sha256'] == final['result_sha256'] == sha(ADMISSION/'result.json',tick)
        and final['completion_sha256'] == sha(ADMISSION/'completion.json',tick)
        and result['context_sha256'] == result['artifacts'][str(ADMISSION/'context.json')] == digest
        and result['source_bindings'] == context['source_bindings'] and result['input_bindings'] == context['input_bindings']
        and result['admission_pid'] == context['admission_pid']
        and 0 <= result['elapsed_s'] <= final['elapsed_s'] < 60 and 0 < result['peak_rss_bytes'] < 2_000_000_000,
        'Actual bounded three-seal source admission')
    require(context['role'] == 'terminal' and context['ranks'] == list(RANKS)
        and context['candidate'] == str(HERE/'build_v1/estimator.py') and context['candidate_sha256'] == CANDIDATE_SHA
        and context['build_sha256'] == BUILD_SHA and context['builder_sha256'] == BUILDER_SHA
        and context['runtime_sha256'] == RUNTIME_SHA and context['source_admission_complete']
        and context['runtime_prerequisite_qualified'] and result['runtime_prerequisite_qualified']
        and context['proof'] == result['proof'] and not context['truth_accessed'] and not context['numerical_imports']
        and not result['truth_accessed'] and not result['numerical_imports'], 'Exact candidate and completed runtime authority')
    require(all(context['source_bindings'].get(k) == v for k,v in reviewed_owners(args).items()), 'Same reviewed endpoint/watch sources')
    bind(context['source_bindings'],tick); bind(context['input_bindings'],tick)
    context['reference_by_rank'] = {int(k):v for k,v in context['reference_by_rank'].items()}
    return context


def run(args):
    source,_ = worker_source(); env = dict(globals())
    exec(compile(source,str(__file__)+':CD_outer_worker_inverse','exec'),env)
    return env['run'](args)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admit',action='store_true'); parser.add_argument('--run',action='store_true')
    parser.add_argument('--role',choices=('terminal',))
    for key in ('source','protocol','watcher','watch-spec','runtime','build','candidate','context'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.admit and not args.run: parser.print_help(); return None
    require(args.admit != args.run and args.role == 'terminal', 'One explicit admission or native stage')
    common = ('source','protocol','watcher','watch-spec')
    required = common + (('runtime','build','candidate') if args.admit else ('context',))
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '') for k in required),
        'Explicit reviewed source and completed stage pins')
    require((args.context_sha256 is None) is args.admit and
        (args.admit or all(getattr(args,k+'_sha256') is None for k in ('runtime','build','candidate'))),
        'Only admission consumes prior receipts; only worker consumes context')
    args.builder_sha256 = BUILDER_SHA
    return args


if __name__ == '__main__':
    _args = arguments()
    if _args is not None: raise SystemExit(admit(_args) if _args.admit else run(_args))
