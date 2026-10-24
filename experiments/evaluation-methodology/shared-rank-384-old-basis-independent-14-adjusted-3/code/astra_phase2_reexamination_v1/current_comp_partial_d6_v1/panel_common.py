"""Target-free independent14 metadata, exact CD consumers and COMP fixtures."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
CD_NATIVE_SHA = 'f5d0e710ce97e292cd3ecce8f7ebcded3fbdc8194a54fbf8e41d5695e61d483a'
ENDPOINT_OWNER_SHA = 'c6ed25eaa1a6ff2131640c5ec665fccb1886446531475bff4d2d83c8076aa652'
COMP_HELPER_SHA = '63832d341b4e4e78b5dda456f0dcfeba42a06114c2142d529320c6dcddea1313'
COMP_OWNERS = {
    'watcher': ('panel_watch.py','324bcd2a43f931f1a38f0f472e99cf593c9a08d8e2f1f63c0ca5ff5ef0d5c904'),
    'worker': ('panel_native.py','2941a78bbddcadb30da28dd29ea3e0976b672f34e2415c48d546a8fc0656dfc5'),
    'admitter': ('panel_admit.py','eeb73f19a84777509a61c3083c25249a77d0d23a22e9a6e5aa96af428d2be9d4'),
    'helper': ('panel_common.py',COMP_HELPER_SHA),
    'scorer': ('panel_score.py','b902977680b39413f57389e19cddd02ab140fe66ff2f727f5650fefb8a7ea7f5'),
    'protocol': ('PANEL_PROTOCOL.md','450d8447493be8ab2844607d96abe31302239b44f9ab9ce9daeee285d25a2f35'),
}


def _load(path, name, digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen metadata owner changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


d = _load(HERE/'native.py','cd_panel_original_native_owner',CD_NATIVE_SHA)
e = _load(COMP/'endpoint_v2.py','cd_panel_original_endpoint_owner',ENDPOINT_OWNER_SHA)
require, sha, read, write, bind, load, splice = d.require,d.sha,d.read,d.write,d.bind,d.load,d.splice
environment, function_source = d.environment,e.function_source
OFFPY, BLAS = d.OFFPY,d.BLAS
ROOT = HERE/'panel_v1'
ADMISSION, OUTPUT, WATCH_OUT = ROOT/'admission',ROOT/'terminal',ROOT/'terminal_watch'
SCORE_OUT, SCORE_WATCH_OUT = ROOT/'score',ROOT/'score_watch'
PUBLIC = (35,100,150,200,250,300,400,500)
VIRGIN = (3016500,3016501,3016502,3016503,3016504,3016505)
RANKS = VALID = PUBLIC+VIRGIN
SEQUENCE = RANKS+(35,)
FIXED = {'terminal':RANKS}
PREFIX = 'current_comp_partial_d6_independent14_'
CONTEXT_SCHEMA, ADMIT_SCHEMA = PREFIX+'context_v1',PREFIX+'admission_v1'
NATIVE_SCHEMA, WATCH_SCHEMA = PREFIX+'native_v1',PREFIX+'watch_v1'
RETIRED_SCHEMA = PREFIX+'admission_watch_v1'
CONTEXT_LIMIT, BUDGET, RSS_LIMIT = 2*1024**2,2**41,8_000_000_000
PROTOCOL_SHA = '84e5c09b79db5cb4878cd2f18a550c6b83a90a25c0ce5b645e76d954a1a7097a'
CANDIDATE, CANDIDATE_SHA = HERE/'build_v1/estimator.py','ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
BUILD_SHA = 'bf17b06afb5e8b7e5b526768c550a7c59eb6ec9808d060cb79b8e1dd083a18af'
BUILDER_SHA = '6ecb36ab4e855e79ea56f66e86940526f44467be9ea240517f34805499d8e7fc'
BUILD_PROTOCOL_SHA = 'bbc2688be5c65d76492a019bf1aeab89ec3d060e03458a8b9078756f044c69d5'
BUILD_WATCH_SHA = '95bae47d29f9767de13eb63b1b88e76ddab11e81ee977af38e4404886b08c53f'
SCORE_SHA = '939f4d1d18172fa732d04f21998a69b498aeb9421db43bb40c8b69c1ca125e81'
SCORE_WATCH_SHA = '9ded35a26256179bca426344f9c0752fcfef7534b8ebbf58041be9ed7e39c3a6'
SCORE40_SOURCES = {
    'score40.py':'b2914608f404909d3325ae61443de93f8c507dba25540a268f6c91b98c8f7dd3',
    'SCORE40_PROTOCOL.md':'9579d3595a74fc9bfc7f4838a8e643605d0f1eb95de7a4ddefc4cfc55469a724',
    'watch_score40.py':'c25a040259498e59989c633104f2f7e1515d8436d6b2a782e95e1ad3e8b63229',
    'SCORE40_WATCH_SPEC.md':'d832513dc560b24e15a7f58a1bdd52fb96d8204179054f4eb328b75c03ba27f4',
    'native40.py':'2d7ed4d25338b199b22919fbcce2ee86140a286d88ab18b27cf6f5ecafadff83',
    'NATIVE40_PROTOCOL.md':'1d894b8517ba8a1e7062595a25437332f69dd98d40e54311e6e7743d6b20072d',
    'watch_native40.py':'547a21000879c0f1a7e26e25b5dec0eb4d38a05bc149de7b490b7e3943e4c6d2',
    'NATIVE40_WATCH_SPEC.md':'25a2591062b9f5b020c766dfbac9ec849ea077db75c3186fdbd0a680ff46ba0f',
}
SCORE40_SEALS = {
    'score40_v1/result.json':SCORE_SHA,
    'score40_v1/completion.json':'c4641e744bdccccacc6107059e99d1438b5237dd34efd0aa3749719851b31d4e',
    'score40_v1/finalization.json':'32560f73b5e132fb16956f49d4241aa8831dac1c3efd872ec124f308498f5e4c',
    'score40_watch_v1/result.json':SCORE_WATCH_SHA,
    'score40_watch_v1/completion.json':'75868e9deeb58f8ac6fc194689c66d2d4bdd3daaca9c2c5c3fa942e3ec5f6d2a',
    'score40_watch_v1/finalization.json':'9adf3144fa0924982044bfc28c8ea1aaf06ff15e19ba74d28b63b5c790ab1b0b',
}
CONTROL = COMP/'panel_v2/terminal'
CONTROL_SCORE, CONTROL_WATCH = COMP/'panel_v2/score',COMP/'panel_v2/terminal_watch'
CONTROL_SHA = 'de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625'
CONTROL_END_SHA = '0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073'
CONTROL_WATCH_SHA = 'cc3502f464e4b77c15398188515a46ff4e421c3267378382ac0d833bf8bdb0ee'
CONTROL_SCORE_SHA = '379f7fa393f81dfe9af0cd30cb9d8ad5a3a08e71f34242c58bc30f540569023c'
CONTROL_SCORE_WATCH_SHA = '99832e1036940913306ad1a2580796d24553a40875988f50c56156eaa771df46'
COMP_SEALS = {
    'admission/context.json':'efd3292e5d9bfacb443dfc7ba4e668bc83c78f90b229168a646132656cd86fe5',
    'terminal/result.json':CONTROL_SHA,'terminal/completion.json':CONTROL_END_SHA,
    'terminal_watch/result.json':CONTROL_WATCH_SHA,
    'terminal_watch/completion.json':'2089674c586ea6ccc758c1435574cb57dc5520d55cb5d8ad8645abcc15673fda',
    'score/result.json':CONTROL_SCORE_SHA,
    'score/completion.json':'c539b369bd5e8ca15ee6213c6acb9e4e208e3fcaeda52fa3d146dbc1d0d3733f',
    'score_watch/result.json':CONTROL_SCORE_WATCH_SHA,
    'score_watch/completion.json':'a87fa8b7d85d7c969971c2807f79c01d1df5df8ba8cbd6aa1892da21707a7c8e',
}
OWNER_FILES = dict(watcher='panel_watch.py',worker='panel_native.py',admitter='panel_admit.py',
    helper='panel_common.py',protocol='PANEL_PROTOCOL.md',watch_spec='PANEL_WATCH_SPEC.md')
_COMP_HELPER = None
_CALLBACKS = None


def comp_helper():
    global _COMP_HELPER
    if _COMP_HELPER is None:
        require(sha(COMP/'panel_common.py') == COMP_HELPER_SHA,'Exact qualified COMP panel metadata')
        original = (COMP/'panel_common.py').read_text()
        source = splice(original,[('import endpoint_v2 as e\n','e = FROZEN_ENDPOINT_OWNER\n')])
        env = dict(__file__=str(COMP/'panel_common.py'),__name__='cd_panel_comp_metadata',FROZEN_ENDPOINT_OWNER=e)
        exec(compile(source,str(COMP/'panel_common.py')+':private-endpoint-import','exec'),env)
        _COMP_HELPER = SimpleNamespace(**env)
    return _COMP_HELPER


pc = comp_helper()
P1, TL, BD = pc.P1,pc.TL,pc.BD
WATCH_OWNER_SHA, NATIVE_OWNER_SHA = pc.WATCH_OWNER_SHA,pc.NATIVE_OWNER_SHA
MONITOR_DIR, MONITOR_SHA = pc.MONITOR_DIR,pc.MONITOR_SHA
MONITOR_TEST_SHA, MONITOR_PROTOCOL_SHA = pc.MONITOR_TEST_SHA,pc.MONITOR_PROTOCOL_SHA
MONITOR_TEST_RESULT_SHA, MONITOR_TEST_END_SHA = pc.MONITOR_TEST_RESULT_SHA,pc.MONITOR_TEST_END_SHA
INVENTORY_SHA = pc.INVENTORY_SHA
EXTRA_LAYOUTS, EXTRA_BINDINGS, PRODUCT_FLOPS, SHAPE = d.EXTRA_LAYOUTS,d.EXTRA_BINDINGS,d.PRODUCT_FLOPS,d.SHAPE
CENSUS, CENSUS_SHA = d.CENSUS,d.CENSUS_SHA
pin_to, inventory, sealed = pc.pin_to,pc.inventory,pc.sealed
transport_admission, panel_modules = pc.transport_admission,pc.panel_modules


def old_args():
    return SimpleNamespace(**{key+'_sha256':digest for key,(_,digest) in COMP_OWNERS.items()},
        context_sha256=COMP_SEALS['admission/context.json'],
        admission_watch_sha256='f7c1e3ce278ce2d45111610ff8828db2ed21f54adce21be3c4237eddae0292bf',
        acquisition_sha256=CONTROL_SHA,acquisition_watch_sha256=CONTROL_WATCH_SHA)


def arguments(role):
    parser = argparse.ArgumentParser(description='Unchanged COMP/partial-D6 independent14 '+role)
    parser.add_argument('--run',action='store_true')
    for key in (*OWNER_FILES,'score40','score40_watch','context','admission_watch'):
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    if role == 'watcher':
        parser.add_argument('--stage',choices=('native',))
        parser.add_argument('--prior-verdict-reviewed',action='store_true')
        parser.add_argument('--numerical-slot-clear',action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return None
    require(role in ('worker','admitter','watcher'),'Only source admission and native acquisition')
    for key in (*OWNER_FILES,'score40','score40_watch'):
        require(re.fullmatch('[0-9a-f]{64}',getattr(args,key+'_sha256') or ''),'Explicit reviewed '+key)
    require(args.protocol_sha256 == PROTOCOL_SHA and args.score40_sha256 == SCORE_SHA
        and args.score40_watch_sha256 == SCORE_WATCH_SHA,'Fixed passed native40 selection, not new choice')
    for key in ('context','admission_watch'):
        value = getattr(args,key+'_sha256')
        require(bool(value) == (role == 'worker') and (value is None or re.fullmatch('[0-9a-f]{64}',value)),
            'Only lightweight worker consumes retired admission')
    if role == 'watcher':
        require(args.stage == 'native' and args.prior_verdict_reviewed and args.numerical_slot_clear,
            'Explicit reviewed verdict and exclusive numerical launch')
    return args


def owners(args,tick=None):
    values = pc.owners(old_args(),tick)
    values.update({str(path):digest for path,digest in d.OWNERS.items()})
    values.update({str(HERE/name):digest for name,digest in SCORE40_SOURCES.items()})
    values.update({str(HERE/'native.py'):CD_NATIVE_SHA,str(HERE/'build.py'):BUILDER_SHA,
        str(HERE/'BUILD_PROTOCOL.md'):BUILD_PROTOCOL_SHA})
    values.update({str(HERE/name):getattr(args,key+'_sha256') for key,name in OWNER_FILES.items()})
    bind(values,tick); return values


def command(args,role,context_sha=None,admission_watch_sha=None):
    require(role in ('admitter','worker'),'No auto-scoring command')
    require(context_sha is None or context_sha == args.context_sha256,'Exact inherited context')
    require(admission_watch_sha is None or admission_watch_sha == args.admission_watch_sha256,'Exact retired handoff')
    cmd = [str(OFFPY),'-B','-u',str(HERE/OWNER_FILES[role]),'--run']
    for key in (*OWNER_FILES,'score40','score40_watch'):
        cmd += ['--'+key.replace('_','-')+'-sha256',getattr(args,key+'_sha256')]
    if role == 'worker':
        cmd += ['--context-sha256',args.context_sha256,'--admission-watch-sha256',args.admission_watch_sha256]
    return cmd


def prerequisites():
    return dict(native40_score_sha256=SCORE_SHA,native40_score_watch_sha256=SCORE_WATCH_SHA,
        control_sha256=CONTROL_SHA,control_watch_sha256=CONTROL_WATCH_SHA,
        control_score_sha256=CONTROL_SCORE_SHA,control_score_watch_sha256=CONTROL_SCORE_WATCH_SHA,
        candidate_sha256=CANDIDATE_SHA,build_sha256=BUILD_SHA)


def seal3(folder,digest,pins,tick=None,limit=1200):
    record,end = sealed(folder,digest,pins,tick=tick)
    path = folder/'finalization.json'; pin_to(pins,tick)(path)
    final = read(path,CONTEXT_LIMIT)
    require(final['passed'] and final['completed'] and final['result_sha256'] == digest
        and final['completion_sha256'] == sha(folder/'completion.json',tick)
        and 0 <= final['elapsed_s'] < limit,'Actual complete new three-seal producer')
    return record,end,final


def metadata_sources():
    original,parent_proof = d.metadata_sources()
    parts = {name:function_source_text(original,name) for name in
        ('validate_diagnostic','after_bill','finish_native','verify_acquisition')}
    finish = splice(parts['finish_native'],[
        ('VALID+(37,)','VALID+(35,)'),
        ("len(result['diagnostic_records']) == 5","len(result['diagnostic_records']) == 15"),
        ('Complete four primaries and repeat37','Complete14 primaries and repeat35'),
        ('records[:4]','records[:14]'),
        ('full_model_calls=5,retired_call_count=5','full_model_calls=15,retired_call_count=15')])
    verify = splice(parts['verify_acquisition'],[
        ('VALID+(37,)','VALID+(35,)'),
        ("record['full_model_calls'] == record['retired_call_count'] == 5",
         "record['full_model_calls'] == record['retired_call_count'] == 15"),
        ("len(record['diagnostic_records']) == 5","len(record['diagnostic_records']) == 15"),
        ('Exact five own prediction/input/diagnostic records','Exact15 own prediction/input/diagnostic records'),
        ("len(paths) == 30, 'Only original25 artifacts plus five small diagnostics'",
         "len(paths) == 90, 'Only original75 artifacts plus15 small diagnostics'")])
    changed = dict(validate_diagnostic=parts['validate_diagnostic'],after_bill=parts['after_bill'],
        finish_native=finish,verify_acquisition=verify)
    proof = dict(CD_parent=parent_proof,functions={name:dict(
        original_sha256=hashlib.sha256(parts[name].encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),inverse_verified=True)
        for name,source in changed.items()},only_roster_count_repeat_changes=True)
    return '\n'.join(changed.values()),proof


def function_source_text(source,name):
    nodes = [node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1,'Unique small source function '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def callbacks():
    global _CALLBACKS
    if _CALLBACKS is None:
        source,_ = metadata_sources(); env = dict(globals(),base=d.base)
        exec(compile(source,str(__file__)+':unchanged-CD-panel-consumers','exec'),env)
        _CALLBACKS = env
    return _CALLBACKS


def after_bill(np,observer,trial,result,folder,refs):
    return callbacks()['after_bill'](np,observer,trial,result,folder,refs)


def finish_function(n):
    require(n is d.base(),'Same lightweight COMP diagnostic owner')
    return callbacks()['finish_native']


def paid_source(n):
    observer,_finish,original,proof = d.prepare(n)
    # This is the exact old COMP panel manifest-to-fixture substitution.
    owner = function_source(COMP/'panel_common.py','paid_source')
    start = owner.index('    start = original.index(')
    stop = owner.index('    proof = dict(',start)
    fixture_code = owner[start:stop]
    env = dict(original=original,splice=splice)
    exec(compile('def apply_fixture():\n'+fixture_code+'    return source\n',
        str(COMP/'panel_common.py')+':same-fixture-splice','exec'),env)
    source = env['apply_fixture']()
    _,metadata_proof = metadata_sources()
    proof = dict(proof,panel_metadata=metadata_proof,panel_fixture_inverse_verified=True,
        panel_fixture_owner_sha256=COMP_HELPER_SHA,
        panel_paid_main_sha256=hashlib.sha256(source.encode()).hexdigest(),
        native_original_main_sha256=proof['paid_loop']['original_paid_main_sha256'],
        no_CHECK4_replay_on_panel_inputs=True)
    return observer,source,proof


def read_context(args,digest,tick=None):
    reviewed = owners(args,tick); pins = {}
    path = ADMISSION/'context.json'
    require(sha(path,tick) == digest,'Explicit retired admission context')
    value = read(path,CONTEXT_LIMIT)
    record,_,_ = seal3(ADMISSION,sha(ADMISSION/'result.json',tick),pins,tick,60)
    require(value['schema'] == CONTEXT_SCHEMA and record['schema'] == ADMIT_SCHEMA
        and value['reviewed_owners'] == record['reviewed_owners'] == reviewed
        and record['context_sha256'] == record['artifacts'][str(path)] == digest
        and record['full_frozen_admission_passed'] and value['full_frozen_admission_passed']
        and record['estimator_executed'] is False and record['numerical_imports'] is False
        and record['truth_accessed'] is False and 0 <= record['elapsed_s'] < 60
        and 0 < record['peak_rss_bytes'] < 2_000_000_000
        and value['ranks'] == list(RANKS) and value['candidate_sha256'] == CANDIDATE_SHA
        and value['build_sha256'] == BUILD_SHA and value['builder_sha256'] == BUILDER_SHA
        and value['build_watch_sha256'] == BUILD_WATCH_SHA and value['prerequisites'] == prerequisites()
        and value['native40_expansion_prerequisite_passed'] and value['COMP_panel_comparator_qualified'],
        'Complete fixed target-free source admission, selected component and actual COMP panel')
    for name in ('source_bindings','input_bindings'):
        require(value[name] == record[name],'Exact admitted inventories')
        bind(value[name],tick)
    require(all(value['source_bindings'].get(k) == v for k,v in reviewed.items()),'All reviewed source owners inherited')
    require(all(value['input_bindings'].get(k) == v for k,v in transport_admission(tick).items()),'Exact monitor mock lineage')
    value['input_bindings'].update(pins); value['input_bindings'][str(path)] = digest
    value['reference_by_rank'] = {int(k):v for k,v in value['reference_by_rank'].items()}
    return value


def retired_admission(args,context,tick=None):
    path = WATCH_OUT/'admission_retired.json'
    require(sha(path,tick) == args.admission_watch_sha256,'Independent retired-admission handoff')
    value = read(path,CONTEXT_LIMIT)
    require(value['schema'] == RETIRED_SCHEMA and value['passed'] and value['completed']
        and value['whole_owned_tree_retired'] and value['context_sha256'] == args.context_sha256
        and value['reviewed_owners'] == owners(args,tick) and value['admission_elapsed_s'] < 60
        and value['whole_elapsed_s'] < 120 and value['maximum_sampled_rss_bytes'] < RSS_LIMIT
        and value['maximum_swap_growth_bytes'] == 0 and value['maximum_sample_interval_s'] <= 1
        and value['monitor_mock_result_sha256'] == MONITOR_TEST_RESULT_SHA
        and value['observational_groups_retired'] and value['foreign_numerical_workers_checked']
        and value['actual_completed_samples'],'Reaped full admission inside original campaign/resource/progress clocks')
    context['input_bindings'][str(path)] = args.admission_watch_sha256
    return value


def verify_native(record,end,digest,args,context,pins=None):
    pins = {} if pins is None else pins
    callbacks()['verify_acquisition'](record,end,digest,args,context)
    require(record['mode'] == 'acquire' and record['fixture_bindings'] == context['fixtures']
        and record['prerequisites'] == context['prerequisites'] and record['admission_watch_sha256'] == args.admission_watch_sha256
        and record['native40_expansion_prerequisite_passed'] and record['all90_artifacts_qualified']
        and record['layer0_byte_exact'] and record['full_admission_in_retired_process']
        and record['truth_files_accessed'] is False and record['no_fit'] and not record['score_measured'],
        'Exact panel inputs, admission, completed40 selection and all15 own outputs')
    inventory(record,pins)
    core = dict(record,artifacts={p:h for p,h in record['artifacts'].items() if not p.endswith('_diagnostics.json')})
    outputs = pc.five_artifact_verifier(context,CANDIDATE,CANDIDATE_SHA,record['initialization_flops'])(
        core,OUTPUT,RANKS,pin_to(pins))
    control,_ = sealed(CONTROL,CONTROL_SHA,pins,CONTROL_END_SHA)
    require(record['versions'] == control['versions'] and record['threadpools'] == control['threadpools'],
        'Same qualified native runtime and actual threadpool frame')
    _,_,v = panel_modules()
    for call,raw in enumerate(outputs):
        prior = control['trials'][call]
        require(raw[:4096] == v.payload(Path(prior['output']),prior['output_bytes_sha256'],pin_to(pins))[:4096],
            'Exact first Gaussian layer for every canonical panel input')
    require(len(record['layer0_checks']) == 15 and all(item == dict(call=i,rank=SEQUENCE[i],byte_exact=True)
        for i,item in enumerate(record['layer0_checks'])),'All15 first-layer parity records')
    pin_to(pins)(OUTPUT/'started.json',record['started_sha256'])
    started = read(OUTPUT/'started.json')
    require(started['source_bindings'] == record['source_bindings'] and started['fixture_bindings'] == context['fixtures']
        and started['candidate_sha256'] == CANDIDATE_SHA and started['nice'] >= 10 and started['no_warmup'],
        'Started/completed immutable ownership and no warmup')
    return outputs


def verify_watch(args,context,pins,tick=None):
    record,_,_ = seal3(WATCH_OUT,args.acquisition_watch_sha256,pins,tick)
    require(record['schema'] == WATCH_SCHEMA and record['stage'] == 'native'
        and all(record.get(k) is True for k in ('sources_unchanged','whole_owned_tree_retired',
            'child_execution_completed','admission_process_reaped_before_native','actual_completed_samples',
            'observational_groups_retired','foreign_numerical_workers_checked'))
        and record['child_result_sha256'] == args.acquisition_sha256
        and record['child_completion_sha256'] == sha(OUTPUT/'completion.json',tick)
        and record['child_finalization_sha256'] == sha(OUTPUT/'finalization.json',tick)
        and record['completed_bill_count'] == 15 and record['admission_context_sha256'] == args.context_sha256
        and record['admission_watch_sha256'] == args.admission_watch_sha256
        and record['prerequisites'] == context['prerequisites'] and record['maximum_sampled_rss_bytes'] < RSS_LIMIT
        and record['maximum_swap_growth_bytes'] == 0 and 0 < record['maximum_sample_interval_s'] <= 1
        and record['monitor_mock_result_sha256'] == MONITOR_TEST_RESULT_SHA
        and record['completed_sample_intervals_s']
        and max(record['completed_sample_intervals_s']) == record['maximum_sample_interval_s']
        and all(0 <= x <= 1 for x in record['completed_sample_intervals_s']) and record['elapsed_s'] < 1200,
        'Independent completed15 acquisition, all three child seals and original resource bounds')
    require(all(record['source_bindings'].get(k) == v for k,v in owners(args,tick).items())
        and all(record['input_bindings'].get(k) == v for k,v in transport_admission(tick).items()),
        'All reviewed panel owners and target-free transport evidence')
    inventory(record,pins,tick)
    return record
