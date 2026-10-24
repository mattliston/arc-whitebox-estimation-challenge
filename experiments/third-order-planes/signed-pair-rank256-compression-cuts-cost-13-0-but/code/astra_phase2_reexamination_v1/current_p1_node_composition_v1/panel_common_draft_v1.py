"""Bounded independent14 metadata context; no model or numerical imports.

Full source construction belongs only to the short-lived panel_admit process.
Native/watch/score rehash its complete dependency inventory, never rebuild P.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ENDPOINT_OWNER_SHA = 'c6ed25eaa1a6ff2131640c5ec665fccb1886446531475bff4d2d83c8076aa652'
if hashlib.sha256((HERE/'endpoint_v2.py').read_bytes()).hexdigest() != ENDPOINT_OWNER_SHA:
    raise RuntimeError('Frozen lightweight helper changed')
import endpoint_v2 as e

NEW, P2, OFFPY = e.NEW, e.P2, e.OFFPY
require, sha, read, write, bind, load, splice = e.require, e.sha, e.read, e.write, e.bind, e.load, e.splice
function_source, environment = e.function_source, e.environment
P1, TL, BD = NEW/'current_d50_native_objective_p1_v1', NEW/'current_d50_terminal_linear_v2', NEW/'birth_direction_fit_v1'
ROOT = HERE/'panel_v2'
ADMISSION, OUTPUT, WATCH_OUT = ROOT/'admission', ROOT/'terminal', ROOT/'terminal_watch'
SCORE_OUT, SCORE_WATCH_OUT = ROOT/'score', ROOT/'score_watch'
PUBLIC = (35,100,150,200,250,300,400,500)
VIRGIN = (3016500,3016501,3016502,3016503,3016504,3016505)
RANKS, SEQUENCE = PUBLIC+VIRGIN, PUBLIC+VIRGIN+(35,)
PROTOCOL_SHA = '450d8447493be8ab2844607d96abe31302239b44f9ab9ce9daeee285d25a2f35'
NATIVE_OWNER_SHA = '63b2ea4edf5fd67f74a30b7fdee1ab3201a87ca13405654fb1ad0d476f0f3e34'
WATCH_OWNER_SHA = '8ffc9ae68388065d49c9e39286893f5da8f5eeabe000db625643fd340442cd28'
ADMIT_OWNER_SHA = 'e8df961a574b27fa49808ba3641fdb68e6b4c37ca18b2ca3f6af04bb65f83bd5'
SCORE_OWNER_SHA = '566f7c255ea7da36aa43a67acab1e3f98cdb5feaad9d46beac6cf14fa0fdcd58'
SCORE_SHA = '17bd92bb65abb5d113cf3cd57df175b865f03295a7e0d96eadfeb133b6658390'
SCORE_END_SHA = '23d4e593af714034e7208a3a5947bef47386cf87a47dd5872c568a22c65415b4'
NATIVE40_SHA = 'a9071e5afa6dbf35e6a6871d6997a7dbc8a1105d2bf3e97bd172c7a1b0e0180f'
NATIVE40_END_SHA = '6c050c04fd565d840273f2a917f2a106182cfed1b69f59a09cd5392497706772'
NATIVE40_WATCH_SHA = '3ef165c3b7f423286d7b10d1ea2af397be602e3417bb4f499e32ed5a49c5b5af'
P1_PANEL_SHA = 'e06d71f9279db2516fa0d45745f0d7b7a9dd37036447445072a17ceeb87a2c11'
P1_WATCH_SHA = '05ca75985c285a1c220690eca3d8884cfb76536cb3708defe7f3d80bdf5e3963'
TL_PANEL_SHA = '0a004bb746a0cdee4345a1b7e5621c2048ef8cfe8e6b46dd2036adb2b581da19'
BD_PANEL_SHA = 'e5f16497f4c8c825f9a203dac3e8c0c3ceac218731605c7c818ae5e8a237268e'
QUALIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
FIXTURE_SHA = '6fcf38e55173a6053c51d536dc9791a0c925a84cd4164d39d159120c0d33acc6'
UTILITY_SHA = '8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304'
CONTROL, CONTROL_SCORE = P1/'panel_v1/terminal', P1/'panel_v1/score'
CONTROL_SHA = 'c8fa1cacfa98d0f45d38235283789cff67c59f32f72fa80a92dd385ed59f8c70'
CONTROL_END_SHA = '35dd7c03bb69c0581d28d1437df4c9c856ee646e9ffa14a81085fb6bc0dd5697'
CONTROL_WATCH_SHA = 'e8d55fa6189be4fc0163c5d305065c3da5cb78a45754a8c923277908808e85da'
CONTROL_SCORE_SHA = '81ebbb54207e4f0060aaa1a91320c17de89b6fdd53e7d5326c17993829e6e526'
CANDIDATE, CANDIDATE_SHA = HERE/'build_v1/terminal_estimator.py', '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
P1_CANDIDATE, P1_SHA = P1/'build_v1/estimator.py', '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
BUILD_SHA = '78f7adbb09ce5749a442c0b4d171110627a0881a478fece4e19f4f5467ec26d6'
PREFIX = 'current_p1_node_composition_independent14_'
CONTEXT_SCHEMA, ADMIT_SCHEMA = PREFIX+'context_v2', PREFIX+'admission_v2'
NATIVE_SCHEMA, WATCH_SCHEMA, SCORE_SCHEMA = PREFIX+'native_v2', PREFIX+'watch_v2', PREFIX+'score_v2'
CONTEXT_LIMIT = 2*1024**2
OWNER_FILES = dict(watcher='panel_watch.py',worker='panel_native.py',admitter='panel_admit.py',
                   helper='panel_common.py',scorer='panel_score.py',protocol='PANEL_PROTOCOL.md')


def arguments(role):
    parser = argparse.ArgumentParser(description='Fixed NODE/P1 independent14 '+role)
    parser.add_argument('--run',action='store_true')
    for key in OWNER_FILES:
        parser.add_argument('--'+key+'-sha256')
    for key in ('context','admission-watch','acquisition','acquisition-watch'):
        parser.add_argument('--'+key+'-sha256')
    if role == 'watcher':
        parser.add_argument('--stage',choices=('native','score'))
        parser.add_argument('--prior-verdict-reviewed',action='store_true')
        parser.add_argument('--numerical-slot-clear',action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return None
    for key in OWNER_FILES:
        require(re.fullmatch('[0-9a-f]{64}',getattr(args,key+'_sha256') or ''), 'Reviewed owner '+key)
    require(args.protocol_sha256 == PROTOCOL_SHA, 'Fixed prospective independent14 protocol')
    score = role == 'scorer' or (role == 'watcher' and args.stage == 'score')
    for key in ('acquisition','acquisition_watch'):
        value = getattr(args,key+'_sha256')
        require(bool(value) == score and (value is None or re.fullmatch('[0-9a-f]{64}',value)), 'Separate score seals')
    for key in ('context','admission_watch'):
        value = getattr(args,key+'_sha256')
        require(bool(value) == (role in ('worker','scorer')) and
                (value is None or re.fullmatch('[0-9a-f]{64}',value)), 'Retired admission handoff only')
    if role == 'watcher':
        require(args.stage in ('native','score') and args.prior_verdict_reviewed and args.numerical_slot_clear,
                'Explicit separate launch and free numerical slot')
    return args


def owners(args,tick=None):
    result = {str(HERE/name):getattr(args,key+'_sha256') for key,name in OWNER_FILES.items()}
    result.update({str(HERE/'endpoint_v2.py'):ENDPOINT_OWNER_SHA,
        str(HERE/'native.py'):e.OLD_NATIVE_SHA,str(HERE/'watch_native.py'):e.OLD_WATCH_SHA,
        str(HERE/'native_v2.py'):NATIVE_OWNER_SHA,str(HERE/'watch_native_v2.py'):WATCH_OWNER_SHA,
        str(HERE/'admit_v2.py'):ADMIT_OWNER_SHA,str(HERE/'ENDPOINT_MEMORY_V2.md'):e.ENDPOINT_SHA,
        str(HERE/'score_v3.py'):SCORE_OWNER_SHA,str(e.HELPER):e.HELPER_SHA,
        str(P1/'panel.py'):P1_PANEL_SHA,str(P1/'watch_panel.py'):P1_WATCH_SHA,
        str(TL/'panel.py'):TL_PANEL_SHA,str(BD/'panel_v2.py'):BD_PANEL_SHA,
        str(BD/'qualify_native_v2.py'):QUALIFIER_SHA,
        str(NEW/'check_response_rows_panel_v1.py'):FIXTURE_SHA,
        str(NEW/'check_response_rows_adjusted_v1.py'):UTILITY_SHA})
    bind(result,tick); return result


def command(args,role,context_sha=None,admission_watch_sha=None):
    require(context_sha is None or context_sha == args.context_sha256, 'Exact inherited context argument')
    require(admission_watch_sha is None or admission_watch_sha == args.admission_watch_sha256,
            'Exact inherited retired-admission argument')
    result = [str(OFFPY),'-B','-u',str(HERE/OWNER_FILES[role]),'--run']
    for key in OWNER_FILES:
        result += ['--'+key+'-sha256',getattr(args,key+'_sha256')]
    if role in ('worker','scorer'):
        result += ['--context-sha256',args.context_sha256,'--admission-watch-sha256',args.admission_watch_sha256]
    if role == 'scorer':
        result += ['--acquisition-sha256',args.acquisition_sha256,
                   '--acquisition-watch-sha256',args.acquisition_watch_sha256]
    return result


def pin_to(pins,tick=None):
    def pin(path,expected=None):
        path = str(Path(path).resolve()); value = sha(path,tick)
        require(expected is None or expected == value, 'Immutable source/input: '+path)
        require(path not in pins or pins[path] == value, 'Conflicting ownership')
        pins[path] = value; return value
    return pin


def inventory(record,pins,tick=None):
    for name in ('source_bindings','input_bindings','artifacts'):
        mapping = record.get(name,{})
        bind(mapping,tick); pins.update(mapping)
    # truth_bindings is deliberately not traversed.


def sealed(folder,digest,pins,end_digest=None,tick=None):
    pin = pin_to(pins,tick)
    pin(folder/'result.json',digest); pin(folder/'completion.json',end_digest)
    record,end = read(folder/'result.json'),read(folder/'completion.json')
    require(all(record.get(k) is True for k in ('passed','completed','hashes_unchanged')) and
            end['passed'] and end['completed'] and end['result_sha256'] == digest, 'Complete immutable seal')
    return record,end


def panel_modules():
    p = load(NEW/'check_response_rows_panel_v1.py','node_panel_original_fixtures',FIXTURE_SHA)
    h = load(NEW/'check_response_rows_adjusted_v1.py','node_panel_original_scalar_utilities',UTILITY_SHA)
    v = load(BD/'qualify_native_v2.py','node_panel_original_lifecycle_verifier',QUALIFIER_SHA)
    require(p.PUBLIC == PUBLIC and p.VIRGIN == VIRGIN and p.RANKS == RANKS, 'Original independent14 population')
    return h,p,v


def paid_source(n):
    owner = load(HERE/'native_v2.py','node_panel_qualified_observer',NATIVE_OWNER_SHA)
    observer,original,proof = owner.prepare(n)
    start = original.index("    manifest=json.loads((STORE/'manifest.json').read_text())")
    stop = original.index('    args.out.mkdir()\n',start)
    source = splice(original,[(original[start:stop],
        "    rows={r['rank']:dict(weights=r['weights_path'],mlp_seed=r['mlp_seed']) for r in PANEL_ROWS}\n"
        "    inputs.update({v['path']:v['sha256'] for v in PANEL_FIXTURES.values()})\n")])
    proof = dict(proof,panel_fixture_inverse_verified=True,
                 panel_paid_main_sha256=hashlib.sha256(source.encode()).hexdigest())
    return observer,source,proof


def finish_function(n):
    source = function_source(HERE/'native.py','finish_native')
    namespace = dict(vars(n),FIXED={'terminal':RANKS})
    exec(compile(source,str(__file__)+':same_NODE_finish','exec'),namespace)
    return namespace['finish_native']


def five_artifact_verifier(context,candidate,candidate_sha,initialization):
    """Original fixture/candidate splice; NODE diagnostics independently verified."""
    h,p,v = panel_modules()
    panel = load(BD/'panel_v2.py','node_panel_verifier_fixture_owner',BD_PANEL_SHA)
    tl = load(TL/'panel.py','node_panel_verifier_owner',TL_PANEL_SHA)
    namespace = dict(vars(tl),CANDIDATE=candidate,CANDIDATE_SHA=candidate_sha)
    # Preserve all original checks; expected startup is supplied by the independent
    # NODE observer/cold-minus-repeat proof, not the old d50-only constant.
    text = function_source(BD/'qualify_native_v2.py','verify_native')
    before = '    expected_init = 1383913666 + (0 if source == BANK else 120)\n'
    after = '    expected_init = EXPECTED_INITIALIZATION\n'
    # Apply the same fixture/candidate derivation, with ONLY startup ownership changed.
    fixture_source = function_source(TL/'panel.py','verifier')
    fixture_source = splice(fixture_source,[(
        "    exec(compile(changed,str(panel.QUALIFIER)+':tl65_panel_verifier','exec'),env)\n",
        "    changed = STARTUP_SPLICE(changed)\n"
        "    env['EXPECTED_INITIALIZATION'] = EXPECTED_INITIALIZATION\n"
        "    exec(compile(changed,str(panel.QUALIFIER)+':tl65_panel_verifier','exec'),env)\n")])
    require(text.count(before) == 1, 'Original fixed initialization check found')
    namespace.update(STARTUP_SPLICE=lambda s:splice(s,[(before,after)]),EXPECTED_INITIALIZATION=initialization)
    exec(compile(fixture_source,str(__file__)+':source_owned_startup_verifier','exec'),namespace)
    return namespace['verifier'](dict(panel=panel,v2=v,fixtures=context['fixtures'],rows=context['rows'],
        tl=SimpleNamespace(BANK=candidate,BANK_SHA=candidate_sha)))


def read_context(args,digest,tick=None):
    reviewed = owners(args,tick); pins = {}
    path = ADMISSION/'context.json'
    require(sha(path,tick) == digest, 'Caller-bound admission context')
    value = read(path,CONTEXT_LIMIT)
    record,end = sealed(ADMISSION,sha(ADMISSION/'result.json',tick),pins,tick=tick)
    require(value['schema'] == CONTEXT_SCHEMA and record['schema'] == ADMIT_SCHEMA and
            value['reviewed_owners'] == record['reviewed_owners'] == reviewed and
            record['context_sha256'] == digest and record['full_frozen_admission_passed'] and
            record['estimator_executed'] is False and record['numerical_imports'] is False and
            record['truth_accessed'] is False and value['ranks'] == list(RANKS) and
            value['candidate_sha256'] == CANDIDATE_SHA and value['build_sha256'] == BUILD_SHA and
            value['prerequisites'] == dict(native40_score_sha256=SCORE_SHA,native40_sha256=NATIVE40_SHA,
                native40_watch_sha256=NATIVE40_WATCH_SHA,control_sha256=CONTROL_SHA,
                control_watch_sha256=CONTROL_WATCH_SHA,control_score_sha256=CONTROL_SCORE_SHA),
            'Fixed fully source-admitted independent14 context')
    bind(value['source_bindings'],tick); bind(value['input_bindings'],tick)
    require(all(value['source_bindings'].get(k) == v for k,v in reviewed.items()), 'All new source owners inherited')
    value['input_bindings'].update(pins); value['input_bindings'][str(path)] = digest
    value['reference_by_rank'] = {int(k):v for k,v in value['reference_by_rank'].items()}
    return value


def retired_admission(args,context,tick=None):
    path = WATCH_OUT/'admission_retired.json'
    require(sha(path,tick) == args.admission_watch_sha256, 'Exact independent admission-retirement seal')
    value = read(path,CONTEXT_LIMIT)
    require(value['schema'] == PREFIX+'admission_watch_v2' and value['passed'] and value['completed'] and
            value['whole_owned_tree_retired'] and value['context_sha256'] == args.context_sha256 and
            value['reviewed_owners'] == owners(args,tick) and value['admission_elapsed_s'] < 60 and
            value['whole_elapsed_s'] < 120 and value['maximum_sampled_rss_bytes'] < 8_000_000_000 and
            value['maximum_swap_growth_bytes'] == 0 and value['maximum_sample_interval_s'] <= 1,
            'Reaped full admission under the same whole/resource/progress clocks')
    context['input_bindings'][str(path)] = args.admission_watch_sha256
    return value


def verify_native(record,end,digest,args,context,pins=None):
    n = e.old(); refs = context['reference_by_rank']
    source = function_source(HERE/'native.py','verify_acquisition')
    changed = splice(source,[("folder = HERE/f'native_{role}_v1'",'folder = PANEL_OUTPUT'),
        ("HERE/'native.py'","HERE/'panel_native.py'"),
        ("HERE/'watch_native.py'","HERE/'panel_watch.py'")])
    namespace = dict(vars(n),SCHEMA=NATIVE_SCHEMA,PANEL_OUTPUT=OUTPUT,FIXED={'terminal':RANKS})
    exec(compile(changed,str(__file__)+':same_NODE_90_artifact_verifier','exec'),namespace)
    namespace['verify_acquisition'](record,end,digest,'terminal',args.worker_sha256,args.watcher_sha256,refs)
    require(record['mode'] == 'acquire' and record['fixture_bindings'] == context['fixtures'] and
            record['proof'] == context['proof'] and record['prerequisites'] == context['prerequisites'] and
            record['admission_context_sha256'] == args.context_sha256 and
            record['admission_watch_sha256'] == args.admission_watch_sha256 and record['layer0_byte_exact'] and
            record['full_admission_in_retired_process'] and record['truth_files_accessed'] is False,
            'Exact fixture/source/handoff ownership')
    pins = {} if pins is None else pins
    inventory(record,pins)
    pin = pin_to(pins)
    core = dict(record,artifacts={p:d for p,d in record['artifacts'].items() if not p.endswith('_diagnostics.json')})
    outputs = five_artifact_verifier(context,CANDIDATE,CANDIDATE_SHA,record['initialization_flops'])(core,OUTPUT,RANKS,pin)
    control,_ = sealed(CONTROL,CONTROL_SHA,pins,CONTROL_END_SHA)
    h,p,v = panel_modules()
    require(record['versions'] == control['versions'] and record['threadpools'] == control['threadpools'],
            'Same qualified native library and actual threadpool frame')
    for call,raw in enumerate(outputs):
        old = control['trials'][call]
        require(raw[:4096] == v.payload(Path(old['output']),old['output_bytes_sha256'],pin)[:4096],
                'Exact unchanged Gaussian first layer on every call')
    pin(OUTPUT/'started.json',record['started_sha256'])
    started = read(OUTPUT/'started.json')
    require(started['source_bindings'] == record['source_bindings'] and started['fixture_bindings'] == context['fixtures']
            and started['candidate_sha256'] == CANDIDATE_SHA and started['nice'] >= 10 and started['no_warmup'],
            'Started and completed source/input ownership')
    return outputs


def verify_watch(args,context,pins,tick=None):
    record,end = sealed(WATCH_OUT,args.acquisition_watch_sha256,pins,tick=tick)
    require(record['schema'] == WATCH_SCHEMA and record['stage'] == 'native' and
            all(record.get(k) is True for k in ('sources_unchanged','whole_owned_tree_retired',
                'child_execution_completed','admission_process_reaped_before_native')) and
            record['child_result_sha256'] == args.acquisition_sha256 and
            record['child_completion_sha256'] == sha(OUTPUT/'completion.json',tick) and
            record['completed_bill_count'] == 15 and record['admission_context_sha256'] == args.context_sha256 and
            record['admission_watch_sha256'] == args.admission_watch_sha256 and
            record['prerequisites'] == context['prerequisites'] and record['maximum_sampled_rss_bytes'] < 8_000_000_000 and
            record['maximum_swap_growth_bytes'] == 0 and record['maximum_sample_interval_s'] <= 1 and
            record['elapsed_s'] < 1200 and all(record['source_bindings'].get(k) == v for k,v in owners(args,tick).items()),
            'All15 calls and independent unchanged resource gates before truths')
    inventory(record,pins,tick)
    return record
