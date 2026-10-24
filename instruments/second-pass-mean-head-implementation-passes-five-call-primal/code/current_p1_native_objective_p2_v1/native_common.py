"""Bounded stdlib native-stage metadata; no historical literal admission here."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
P1, TL, BD = (NEW/name for name in ('current_d50_native_objective_p1_v1',
    'current_d50_terminal_linear_v2','birth_direction_fit_v1'))
OFFPY = P2/'data_gate/.venv/bin/python'
PROTOCOL_SHA = 'b2ceb8ac16086e19a5630658df7fb43afaa17b47d761267a1d84ffadce8d9b16'
P1_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
BUILDER_SHA = 'dce1494da87a32876f9e2e29d06f4b28bbc807c65f06b9db33321512aa66ce9c'
TRAIN_COMMON_SHA = 'f443ec27b2d9d1547d2ba755431226d4ac1d7ecba73fdd0e74b18dfd8434e1f9'
TRAIN_WORKER_SHA = '41cdc5993e60510f86401971f558bc370400027d08ba1f8295831be37e7b21a8'
TRAIN_WATCH_SHA = 'fe227690050e88265c972f67f14732578f1bb55aef73f58489f118d67ee5640d'
P1_NATIVE_SHA = 'fad59d76d82baf40c823e0468b7304a909b719f8bc4a8c1f5c4e7118a4f9d9e8'
P1_NATIVE_OLD_SHA = 'ebab249a8a1bdd0c2017a1a24736c98eac708310865f50588d7ab4043170e2d4'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
OBSERVER_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
OBSERVER_COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
TL_NATIVE_SHA = '69c8eeee8078feb2b12662f128ea873e5d67435182d4c30bd551a1369c84f69c'
VERIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
RESOURCE = TL/'watch_fit.py'
RESOURCE_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
CENSUS_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
FIXED = {'null':(35,100),'terminal':VALID}
BUDGET, INITIALIZATION, CONTEXT_LIMIT = 2**41,1383913786,8*1024**2
SCHEMA = 'current_p1_native_objective_p2_native_v1'
WATCH_SCHEMA = 'current_p1_native_objective_p2_native_watch_v1'
ADMIT_SCHEMA = 'current_p1_native_objective_p2_native_admission_v1'
CONTEXT_SCHEMA = 'current_p1_native_objective_p2_native_context_v1'
HANDOFF_SCHEMA = 'current_p1_native_objective_p2_native_admission_watch_v1'
OWNER_FILES = dict(watcher='native_watch.py',worker='native_capture.py',admitter='native_admit.py',
                   helper='native_common.py',endpoint='NATIVE_ENDPOINT.md')
FUTURE = ('build','candidate','training','training-watch','qualification','qualification-watch',
          'terminal-payload','terminal-seal','validation-seal')
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
REFERENCES = {
    'null':(P1/'panel_v1/terminal','c8fa1cacfa98d0f45d38235283789cff67c59f32f72fa80a92dd385ed59f8c70',
            '35dd7c03bb69c0581d28d1437df4c9c856ee646e9ffa14a81085fb6bc0dd5697'),
    'terminal':(P1/'native2_terminal_v1','7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3',
                'b73a957bf78e45da0db2741cea0b33a2cd98317726edaaa0954eaece054e5957')}


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'Target-array access forbidden')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick: tick()
    return digest.hexdigest()


def read(path, bound=None):
    path = Path(path)
    require(not path.is_symlink() and (bound is None or path.stat().st_size <= bound), 'Bounded regular JSON')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def put(mapping, path, digest):
    path = str(Path(path).resolve())
    require(re.fullmatch('[0-9a-f]{64}',digest or '') and
            (path not in mapping or mapping[path] == digest), 'Unambiguous flat hash binding')
    mapping[path] = digest


def bind(mapping, tick=None):
    for path,digest in mapping.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or '')
                and sha(path,tick) == digest, 'Immutable dependency: '+str(path))


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed source owner')
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def function_source(path, name):
    text = Path(path).read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name]
    require(len(nodes) == 1, 'Unique frozen source object')
    node = nodes[0]
    return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def splice(source, edits):
    changed = source
    for old,new in edits:
        require(changed.count(old) == 1, 'Unique admission/metadata source substitution')
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique source inverse')
        inverse = inverse.replace(new,old)
    require(inverse == source, 'Exact complete source inverse')
    return changed


def paths(role):
    require(role in FIXED, 'Fixed native role')
    return (HERE/f'native_{role}_admission_v1',HERE/f'native_{role}_v1',HERE/f'native_{role}_watch_v1')


def arguments(kind):
    parser = argparse.ArgumentParser(description='P2 actual-native '+kind+'; inert without --run')
    parser.add_argument('--run',action='store_true'); parser.add_argument('--role',choices=tuple(FIXED))
    for key in (*OWNER_FILES,'protocol',*FUTURE,'null-result','null-watch','context','admission-watch'):
        parser.add_argument('--'+key+'-sha256')
    if kind == 'watcher':
        parser.add_argument('--prior-verdict-reviewed',action='store_true')
        parser.add_argument('--numerical-slot-clear',action='store_true')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return None
    require(args.role in FIXED and args.protocol_sha256 == PROTOCOL_SHA, 'Fixed role and unchanged protocol')
    for key in (*OWNER_FILES,'protocol',*FUTURE):
        require(re.fullmatch('[0-9a-f]{64}',getattr(args,key.replace('-','_')+'_sha256') or ''), 'Every reviewed/future SHA required')
    for key in ('null_result','null_watch'):
        value = getattr(args,key+'_sha256')
        require(bool(re.fullmatch('[0-9a-f]{64}',value or '')) if args.role == 'terminal' else value is None,
                'Only terminal requires the completed null and watch')
    for key in ('context','admission_watch'):
        value = getattr(args,key+'_sha256')
        require(bool(re.fullmatch('[0-9a-f]{64}',value or '')) if kind == 'worker' else value is None,
                'Only the native child accepts the parent-pinned context/handoff')
    return args


def requested(args):
    return {key.replace('-','_')+'_sha256':getattr(args,key.replace('-','_')+'_sha256')
            for key in ('protocol',*FUTURE,'null-result','null-watch')}


def owners(args, tick=None):
    pins = {str(HERE/name):getattr(args,key+'_sha256') for key,name in OWNER_FILES.items()}
    pins.update({str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'build.py'):BUILDER_SHA,
        str(HERE/'common.py'):TRAIN_COMMON_SHA,str(HERE/'worker.py'):TRAIN_WORKER_SHA,str(HERE/'watch.py'):TRAIN_WATCH_SHA,
        str(P1/'native_v2.py'):P1_NATIVE_SHA,str(P1/'native.py'):P1_NATIVE_OLD_SHA,
        str(TL/'native_validation.py'):TL_NATIVE_SHA,str(TL/'native.py'):OBSERVER_SHA,
        str(TL/'common.py'):OBSERVER_COMMON_SHA,str(BD/'native_v2.py'):NATIVE_SHA,
        str(BD/'qualify_native_v2.py'):VERIFIER_SHA,str(RESOURCE):RESOURCE_SHA,str(CENSUS):CENSUS_SHA})
    bind(pins,tick); return pins


def environment():
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
            and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'OFFPY/nice10/BLAS1/no fallback')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')), 'Pre-numerical stdlib stage')


def command(args, kind, context=None, handoff=None):
    cmd = [str(OFFPY),'-B','-u',str(HERE/OWNER_FILES[kind]),'--run','--role',args.role]
    for key in (*OWNER_FILES,'protocol',*FUTURE,'null-result','null-watch'):
        value = getattr(args,key.replace('-','_')+'_sha256')
        if value is not None: cmd += ['--'+key+'-sha256',value]
    if context: cmd += ['--context-sha256',context,'--admission-watch-sha256',handoff]
    return cmd


def sealed(folder, digest, pin, end_sha=None):
    pin(folder/'result.json',digest); pin(folder/'completion.json',end_sha)
    record,end = read(folder/'result.json'),read(folder/'completion.json')
    require(all(record.get(k) is True for k in ('passed','completed','hashes_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == digest, 'Completed immutable PASS')
    return record,end


def reference(role, pin):
    folder,digest,end_sha = REFERENCES[role]
    record,_ = sealed(folder,digest,pin,end_sha)
    require(record['candidate_sha256'] == P1_SHA and record['truth_accessed'] is False
            and all(record[k] for k in ('all16_domains_qualified','all47_selectors_qualified','own_complete_bills'))
            and record['source_bindings'][str(STORE/'manifest.json')] == STORE_SHA, 'Qualified retained P1 comparator')
    for path,value in record['artifacts'].items(): pin(path,value)
    refs = {}
    for trial,prepared,domain in zip(record['trials'],record['input_preparation'],record['domain_diagnostics']):
        rank,call = trial['rank'],trial['call']
        require(prepared['rank'] == domain['rank'] == rank and prepared['call'] == domain['call'] == call, 'Reference call identity')
        if rank in FIXED[role] and rank not in refs:
            require(record['artifacts'][trial['output']] == trial['output_sha256'], 'Reference output seal')
            refs[rank] = dict(trial=trial,weight_file_sha256=prepared['weight_file_sha256'],domain=domain)
    require(set(refs) == set(FIXED[role]), 'Every fixed reference rank')
    if role == 'null': require([refs[r]['trial']['call'] for r in (35,100)] == [0,1], 'Matching initial call history')
    return refs


def verifier(candidate, digest, pin):
    pin(BD/'qualify_native_v2.py',VERIFIER_SHA)
    base = load(BD/'qualify_native_v2.py','p2_native_original_artifact_verifier',VERIFIER_SHA)
    original = function_source(BD/'qualify_native_v2.py','verify_native')
    before = ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
              "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)")
    changed = splice(original,[(before,'    expected_sha = ELIGIBLE_SOURCES.get(source)')])
    env = dict(vars(base),ELIGIBLE_SOURCES={Path(candidate):digest,P1/'build_v1/estimator.py':P1_SHA})
    exec(compile(changed,str(__file__)+':original_artifact_verifier','exec'),env)
    return env['verify_native']


def validate_domains(record):
    sequence = list(record['ranks'])+[record['ranks'][0]]
    require(len(record['domain_diagnostics']) == len(sequence), 'Every original passive domain packet')
    cells = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
    for call,(rank,domain) in enumerate(zip(sequence,record['domain_diagnostics'])):
        require(domain['call'] == call and domain['rank'] == rank and domain['variance_count'] == 16
                and domain['selector_count'] == 47 and set(domain['variance_domain']) == {str(l) for l in range(16)}
                and set(domain['selection_cells']) == cells, 'All16/all47 domain scope')
        require(all(v['finite'] and v['minimum'] >= 1e-10 and v['below_floor_count'] == 0
                    for v in domain['variance_domain'].values()), 'Every unfloored finite prevariance')
        require(all(v['floor'] == .01 and 0 < v['count'] <= 1024 and v['count'] % 32 == 0
                    and all(re.fullmatch('[0-9a-f]{64}',v[k]) for k in ('indices_sha256','mask_sha256'))
                    for v in domain['selection_cells'].values()), 'Original selector floors/counts/hashes')
    require({k:v for k,v in record['domain_diagnostics'][0].items() if k != 'call'} ==
            {k:v for k,v in record['domain_diagnostics'][-1].items() if k != 'call'}, 'Own repeated domain bytes')


def verify_native(record, end, digest, args, context, pin):
    role = context['role']; candidate = Path(context['candidate'])
    require(record['schema'] == SCHEMA and record['stage'] == 'native' and record['role'] == role
            and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                'all16_domains_qualified','all47_selectors_qualified','own_complete_bills',
                'paid_arithmetic_and_lifecycle_unchanged','observer_event_filter_inverse_verified',
                'all_outputs_and_bills_sealed_before_scoring'))
            and end['passed'] and end['completed'] and end['result_sha256'] == digest
            and record['truth_accessed'] is False and record['prerequisites'] == context['prerequisites']
            and record['candidate_sha256'] == context['candidate_sha256']
            and record['observer_proof'] == context['proof'] and record['full_admission_in_retired_process'], 'Same admitted native acquisition')
    for path,value in context['reviewed_owners'].items():
        require(record['source_bindings'].get(path) == value, 'Every executed native owner bound')
    outputs = verifier(candidate,context['candidate_sha256'],pin)(record,paths(role)[1],FIXED[role],pin)
    validate_domains(record)
    require(record['initialization_flops'] == INITIALIZATION, 'Actual complete startup ownership')
    if role == 'null':
        require(record['null_same_history_first_two_full_parity'] and record['null_full_output_bill_parity']
                and record['unequal_history_repeat_opcount_compared'] is False, 'Exact declared P1 null scope')
        for call,trial in enumerate(record['trials']):
            ref = context['reference_by_rank'][str(trial['rank'])]
            require(trial['output_bytes_sha256'] == ref['trial']['output_bytes_sha256'] and
                trial['flops'] == ref['trial']['flops']-(INITIALIZATION if call == 2 else 0), 'Null all16 and own complete F')
            if call < 2: require(trial['operation_count'] == ref['trial']['operation_count'], 'Matching initial opcount history')
            require({k:v for k,v in record['domain_diagnostics'][call].items() if k != 'call'} ==
                    {k:v for k,v in ref['domain'].items() if k != 'call'}, 'P1 reference domain/selector bytes')
    return outputs


def read_context(args, digest, tick=None):
    admission,_,_ = paths(args.role); expected = owners(args,tick)
    jp,rp,cp = (admission/name for name in ('context.json','result.json','completion.json'))
    require(sha(jp,tick) == digest, 'Parent-pinned admitted context')
    context,record,end = read(jp,CONTEXT_LIMIT),read(rp,CONTEXT_LIMIT),read(cp,CONTEXT_LIMIT)
    require(context['schema'] == CONTEXT_SCHEMA and record['schema'] == ADMIT_SCHEMA
            and context['role'] == args.role and context['ranks'] == list(FIXED[args.role])
            and context['requested_seals'] == requested(args) and context['reviewed_owners'] == expected
            and record['reviewed_owners'] == expected and record['context_sha256'] == digest
            and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == sha(rp,tick)
            and context['full_frozen_admission_passed'] and context['observer_inverse_verified']
            and context['paid_loop_inverse_verified'] and record['numerical_imports'] is False
            and record['estimator_executed'] is False and record['truth_accessed'] is False, 'Exact completed full-admission owner')
    require(context['candidate_sha256'] == (P1_SHA if args.role == 'null' else args.candidate_sha256)
            and Path(context['candidate']) == HERE/'build_v1'/('null_estimator.py' if args.role == 'null' else 'estimator.py'), 'Sole selected literal')
    for group in ('source_bindings','input_bindings'): bind(context[group],tick)
    require(all(context['source_bindings'].get(k) == v for k,v in expected.items()), 'All reviewed sources admitted')
    for path in (jp,rp,cp): put(context['input_bindings'],path,sha(path,tick))
    return context


def retired_admission(args, context):
    path = paths(args.role)[2]/'admission_retired.json'
    require(sha(path) == args.admission_watch_sha256, 'Exact independent admission-retirement seal')
    value = read(path,CONTEXT_LIMIT)
    require(value['schema'] == HANDOFF_SCHEMA and value['role'] == args.role and value['passed']
            and value['completed'] and value['whole_owned_tree_retired'] and value['context_sha256'] == args.context_sha256
            and value['reviewed_owners'] == owners(args) and value['admission_elapsed_s'] < 60
            and value['whole_elapsed_s'] < 120 and value['maximum_sampled_rss_bytes'] < 8_000_000_000
            and value['maximum_swap_growth_bytes'] == 0 and value['maximum_sample_interval_s'] <= 1, 'Bounded retired full-admission process')
    put(context['input_bindings'],path,args.admission_watch_sha256)
    return value
