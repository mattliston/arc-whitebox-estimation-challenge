"""Terminal-only CHECK4: source-inverse JOINT loop, original COMP observer."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
JOINT = NEW/'current_comp_joint_p_node_v1/native.py'
JOINT_SHA = '851813e9679cd58f1b23bbd15de36cdeb866df0e646e1e8cb9e9aa86e924b0bc'
MODEL_SHA = 'f5c26f04750fa147ef897484ddc3a152ad744e6170b7542e8046af3193a79656'
PROPOSAL_PROTOCOL_SHA = '0a1db21c3a9f6f0a268ae8c47c1f4ad1c6b16ff6a0438aaeb2278dfaa6e97359'
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
SCHEMA = 'current_comp_cstar_proposal_native_v1'
ADMISSION_SCHEMA = 'current_comp_cstar_proposal_native_admission_v1'
BUILD_WATCH_SCHEMA = 'current_comp_cstar_proposal_build_watch_v1'
RANKS = (37,39,42,48)
OUTPUT = HERE/'native_terminal_v1'
ADMISSION = HERE/'native_terminal_admission_v1'
EXTRA_OWNERS = {
    JOINT:JOINT_SHA,
    HERE/'model.py':MODEL_SHA,
    HERE/'PROTOCOL.md':PROPOSAL_PROTOCOL_SHA,
    COMP/'build_v1/terminal_estimator.py':PARENT_SHA,
    COMP/'native_terminal_watch_v2/result.json':'3ef165c3b7f423286d7b10d1ea2af397be602e3417bb4f499e32ed5a49c5b5af',
    COMP/'native_terminal_watch_v2/completion.json':'ee7204fb3e444a1194d00bc43f6418e3119c965fac4b2630afea2c7578483555',
}
HASH_KEYS = ('source','model','protocol','watcher','watch_spec','builder','build_protocol',
    'build','candidate','build_watch','build_watcher','build_watch_spec')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def strict_sha(path,tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),'No target-array access')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick:
                tick()
    return digest.hexdigest()


def _splice(source,changes):
    result = source
    for before,after in changes:
        require(result.count(before) == 1,'Unique inherited native span: '+before[:100])
        result = result.replace(before,after)
    inverse = result
    for before,after in reversed(changes):
        require(inverse.count(after) == 1,'Unique inherited native inverse: '+after[:100])
        inverse = inverse.replace(after,before)
    require(inverse == source,'Whole frozen JOINT native source restored')
    ast.parse(result)
    return result


def _merge(target,mapping):
    require(isinstance(mapping,dict),'Flat target-free binding map')
    for path,digest in mapping.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or '')
            and (path not in target or target[path] == digest) and strict_sha(path) == digest,
            'Exact nonconflicting source/input ownership')
        target[path] = digest


def extra_sources(a):
    return {**EXTRA_OWNERS,HERE/'NATIVE_WATCH_SPEC.md':a.watch_spec_sha256,
        HERE/'watch_build.py':a.build_watcher_sha256,
        HERE/'BUILD_WATCH_SPEC.md':a.build_watch_spec_sha256}


def build_watch_admission(a,context):
    """Completed external build supervision, never a future receipt guess."""
    folder = HERE/'build_watch_v1'
    rp,cp,fp = (folder/name for name in ('result.json','completion.json','finalization.json'))
    require(strict_sha(rp) == a.build_watch_sha256,'Explicit completed build-watch digest')
    record,end,final = (read(path) for path in (rp,cp,fp))
    flags = ('passed','completed','hashes_unchanged','sources_unchanged','whole_owned_tree_retired',
             'cleanup_passed','observational_groups_retired')
    require(record['schema'] == BUILD_WATCH_SCHEMA and all(record.get(k) is True for k in flags)
        and end['passed'] and end['completed'] and final['passed'] and final['completed']
        and end['result_sha256'] == final['result_sha256'] == a.build_watch_sha256
        and final['completion_sha256'] == strict_sha(cp)
        and record['child_result_sha256'] == a.build_sha256
        and record['child_completion_sha256'] == strict_sha(HERE/'build_v1/completion.json')
        and record['child_finalization_sha256'] == strict_sha(HERE/'build_v1/finalization.json')
        and 0 <= record['elapsed_s'] < 120 and 0 <= final['elapsed_s'] < 120
        and 0 < record['maximum_combined_rss_bytes'] < 2_000_000_000
        and 0 < record['maximum_interval_s'] <= 1 and record['maximum_swap_growth_bytes'] == 0,
        'Actual complete bounded build and external three-seal ownership')
    pins = {str(path):strict_sha(path) for path in (rp,cp,fp)}
    for key in ('source_bindings','input_bindings','artifacts'):
        _merge(pins,record.get(key,{}))
    expected = {str(HERE/'build.py'):a.builder_sha256,str(HERE/'BUILD_PROTOCOL.md'):a.build_protocol_sha256,
        str(HERE/'watch_build.py'):a.build_watcher_sha256,
        str(HERE/'BUILD_WATCH_SPEC.md'):a.build_watch_spec_sha256}
    require(all(pins.get(k) == v for k,v in expected.items()),'Reviewed build/supervisor source maps')
    candidate = context['candidates']['terminal']
    parent = context['candidates']['parent']
    require(Path(candidate['path']) == HERE/'build_v1/estimator.py' and candidate['sha256'] == a.candidate_sha256
        and parent['sha256'] == PARENT_SHA and strict_sha(candidate['path']) == a.candidate_sha256,
        'Only the source-inverse cstar terminal of unchanged COMP')
    return pins


def control_watch_admission(baseline,end,inputs):
    """Current COMP has actual two-seal native/watch receipts; no invented third."""
    folder = COMP/'native_terminal_watch_v2'
    record,completion = (read(folder/name) for name in ('result.json','completion.json'))
    require(baseline['schema'] == 'current_p1_node_composition_native_v2'
        and baseline['candidate_sha256'] == PARENT_SHA and baseline['role'] == 'terminal'
        and baseline['truth_accessed'] is False and baseline['hashes_unchanged'] and baseline['sources_unchanged']
        and len(baseline['trials']) == len(baseline['input_preparation']) == len(baseline['diagnostic_records']) == 41
        and [x['rank'] for x in baseline['trials'][:4]] == list(RANKS)
        and baseline['trials'][-1]['rank'] == 37 and end['completed']
        and record['schema'] == 'current_p1_node_composition_native_watch_v2'
        and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'whole_owned_tree_retired','child_execution_completed'))
        and completion['passed'] and completion['completed']
        and completion['result_sha256'] == EXTRA_OWNERS[folder/'result.json']
        and record['child_result_sha256'] == OWNERS[COMP/'native_terminal_v2/result.json']
        and record['child_completion_sha256'] == OWNERS[COMP/'native_terminal_v2/completion.json']
        and record['completed_bill_count'] == 41 and 0 <= record['elapsed_s'] < 1900
        and 0 < record['maximum_sampled_rss_bytes'] < 8_000_000_000
        and record['maximum_swap_growth_bytes'] == 0 and 0 < record['maximum_sample_interval_s'] <= 1,
        'Exact retained COMP40, same first-four physical inputs/cold history')
    for path in (folder/'result.json',folder/'completion.json'):
        inputs[str(path)] = strict_sha(path)


def coefficient_startup_checks(record):
    for call,item in enumerate(record['diagnostic_records']):
        startup = read(item['path'])['startup']
        calls = [x for x in startup['materializations'] if x['name'] == '_bd_coefficients']
        require(len(calls) == 2 and all(x['literal_elements'] == 60 for x in calls)
            and [x['inside_prefix'] for x in calls] == [True,False]
            and [x['flops'] for x in calls] == ([120,0] if call == 0 else [0,0]),
            'Both existing cstar getters, exact120F cold/0 warm, nested once in prefix')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--admit',action='store_true'); stages.add_argument('--run',action='store_true')
    parser.add_argument('--role',choices=('terminal',))
    for key in (*HASH_KEYS,'context'):
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args = parser.parse_args()
    if not args.run and not args.admit:
        parser.print_help(); return None
    require(args.role == 'terminal','One fixed terminal, no zero/alternate campaign')
    for key in HASH_KEYS:
        require(re.fullmatch('[0-9a-f]{64}',getattr(args,key+'_sha256') or ''),'Explicit reviewed '+key)
    require(args.model_sha256 == MODEL_SHA,'Exact qualified proposal model owner; never imported here')
    require((args.admit and args.context_sha256 is None) or (args.run and
        re.fullmatch('[0-9a-f]{64}',args.context_sha256 or '')),'Only worker consumes retired context')
    # Internal compatibility label only; no scientific arm selection.
    args.arm = 'terminal'
    return args


require(strict_sha(JOINT) == JOINT_SHA,'Frozen qualified CHECK4 harness')
_original = JOINT.read_text()
_changes = [
    ('def arguments():','def _unused_original_arguments():'),
    ("env=dict(vars(n),FIXED={k:RANKS for k in ('p_only','joint')})",
     "env=dict(vars(n),FIXED={'terminal':RANKS})"),
    ("HERE/'watch_native.py':a.watcher_sha256,HERE/'build.py':a.builder_sha256,HERE/'BUILD_PROTOCOL.md':a.build_protocol_sha256}",
     "HERE/'watch_native.py':a.watcher_sha256,HERE/'build.py':a.builder_sha256,HERE/'BUILD_PROTOCOL.md':a.build_protocol_sha256,**extra_sources(a)}"),
    ("builder=module(HERE/'build.py','joint_native_source_builder')",
     "builder=module(HERE/'build.py','cprop_native_source_builder')"),
    ('context=builder.verify_build(a.arm,a.build_sha256,a.candidate_sha256,a.builder_sha256,a.build_protocol_sha256)',
     'context=builder.verify_build(a.build_sha256,a.candidate_sha256,a.builder_sha256,a.build_protocol_sha256)'),
    ("        inputs=dict(context['input_bindings'])\n",
     "        inputs=dict(context['input_bindings']);_merge(inputs,build_watch_admission(a,context))\n"),
    ("        refs={}\n", "        control_watch_admission(baseline,end,inputs)\n        refs={}\n"),
    ("candidate=HERE/f'build_{a.arm}_v1/estimator.py';",
     "candidate=HERE/'build_v1/estimator.py';"),
    ("        document=dict(arm=a.arm,ranks=list(RANKS),candidate=str(candidate),candidate_sha256=a.candidate_sha256,\n",
     "        document=dict(schema=ADMISSION_SCHEMA,role='terminal',arm=a.arm,ranks=list(RANKS),candidate=str(candidate),candidate_sha256=a.candidate_sha256,\n"
     "            build_watch_sha256=a.build_watch_sha256,builder_sha256=a.builder_sha256,\n"
     "            requested_seals=vars(a),native_adapter_source_inverse=NATIVE_ADAPTER_PROOF,\n"),
    ("require(len(json.dumps(document).encode())<4_000_000,'Bounded metadata-only admission context')",
     "require(len(json.dumps(document).encode())<2*1024**2,'Bounded metadata-only admission context')"),
    ("        result.update(passed=True,completed=True,context_sha256=sha(folder/'context.json'),sources_unchanged=True)\n",
     "        result.update(passed=True,completed=True,context_sha256=sha(folder/'context.json'),sources_unchanged=True,\n"
     "            hashes_unchanged=True,schema=ADMISSION_SCHEMA,role='terminal',source_bindings=sources,input_bindings=inputs,\n"
     "            artifacts={str(folder/'context.json'):sha(folder/'context.json')})\n"),
    ("        and context['source_bindings'][str(HERE/'watch_native.py')]==a.watcher_sha256,'Exact reviewed numerical/supervisor owners')",
     "        and context['source_bindings'][str(HERE/'watch_native.py')]==a.watcher_sha256\n"
     "        and all(context['source_bindings'].get(str(p))==h for p,h in extra_sources(a).items())\n"
     "        and context['build_watch_sha256']==a.build_watch_sha256\n"
     "        and context['builder_sha256']==a.builder_sha256\n"
     "        and context['build_sha256']==a.build_sha256 and context['candidate_sha256']==a.candidate_sha256\n"
     "        and context['source_bindings'].get(str(HERE/'build.py'))==a.builder_sha256\n"
     "        and context['source_bindings'].get(str(HERE/'BUILD_PROTOCOL.md'))==a.build_protocol_sha256\n"
     "        and context['native_adapter_source_inverse']==NATIVE_ADAPTER_PROOF,'Exact reviewed numerical/supervisor owners')"),
    ('        finish(result);result.update(full_model_calls=5,retired_call_count=5)\n',
     '        finish(result);coefficient_startup_checks(result)\n'
     '        result.update(full_model_calls=5,retired_call_count=5,cstar_materialization_owned=True)\n'),
    ("header=dict(schema='current_comp_joint_p_node_native_v1',stage='native',role=a.arm,arm=a.arm,completed=False,",
     "header=dict(schema=SCHEMA,stage='native',role=a.arm,arm=a.arm,completed=False,\n"
     "        builder_sha256=a.builder_sha256,build_watch_sha256=a.build_watch_sha256,\n"
     "        native_adapter_source_inverse=NATIVE_ADAPTER_PROOF,coefficient_literal_only=True,\n"
     "        exact_COMP_control_qualified=True,no_D6_composition=True,native_gradient_claim=False,\n"),
    ("if __name__=='__main__':\n    a=arguments()\n    if a is not None:raise SystemExit(admit(a) if a.admit else run(a))\n",
     "def _inherited_entrypoint():\n    a=arguments()\n    if a is None:return 0\n    return admit(a) if a.admit else run(a)\n"),
]
_source = _splice(_original,_changes)
NATIVE_ADAPTER_PROOF = dict(owner=str(JOINT),owner_sha256=JOINT_SHA,
    original_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_sha256=hashlib.sha256(_source.encode()).hexdigest(),whole_source_inverse_verified=True,
    paid_loop_and_original_COMP_observer_unchanged=True,no_new_observed_code_or_array_operations=True,
    changes='terminal-only authority/build/control metadata; no participant arithmetic')
exec(compile(_source,str(JOINT)+':terminal-cstar-metadata-only','exec'),globals())
# Strengthen the inherited unqualified SHA helper without touching numerical code.
sha = strict_sha
OWNERS.update(EXTRA_OWNERS)


def verify_acquisition(record,end,digest,args,context,tick=None):
    """The original COMP saved-file verifier with only current metadata owners."""
    path = COMP/'native.py'
    require(sha(path,tick) == OWNERS[path],'Exact COMP native metadata verifier')
    original = extract(path,'verify_acquisition')
    source = splice(original,[("str(HERE/'PROTOCOL.md')", "str(HERE/'NATIVE_PROTOCOL.md')")])
    n = module(path,'cprop_saved_file_metadata_owner')
    namespace = dict(vars(n),HERE=HERE,FIXED={'terminal':RANKS},SCHEMA=SCHEMA,
        CANDIDATES={'terminal':context['candidate_sha256']},BUILD_SHA=context['build_sha256'],
        BUILDER_SHA=context['builder_sha256'],PROTOCOL_SHA=args.protocol_sha256,
        sha=lambda p:strict_sha(p,tick))
    # Unchanged private bind body resolves the stage-local ticking SHA helper.
    exec(compile(extract(path,'bind'),str(path)+':unchanged-private-binding','exec'),namespace)
    exec(compile(source,str(path)+':same-COMP-complete-file-verifier','exec'),namespace)
    refs = {int(k):v for k,v in context['reference_by_rank'].items()}
    namespace['verify_acquisition'](record,end,digest,'terminal',args.source_sha256,args.watcher_sha256,refs)
    require(record['full_model_calls'] == record['retired_call_count'] == 5
        and record['build_watch_sha256'] == args.build_watch_sha256 == context['build_watch_sha256']
        and record['coefficient_literal_only'] and record['exact_COMP_control_qualified']
        and record['no_D6_composition'] and record['native_gradient_claim'] is False
        and record['native_adapter_source_inverse'] == NATIVE_ADAPTER_PROOF
        and record['cstar_materialization_owned'],
        'Actual sole coefficient endpoint and exact five-call source ownership')
    coefficient_startup_checks(record)
    if tick:
        tick()
    return dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),whole_source_inverse_verified=True)


verify_native = verify_acquisition


if __name__ == '__main__':
    raise SystemExit(_inherited_entrypoint())
