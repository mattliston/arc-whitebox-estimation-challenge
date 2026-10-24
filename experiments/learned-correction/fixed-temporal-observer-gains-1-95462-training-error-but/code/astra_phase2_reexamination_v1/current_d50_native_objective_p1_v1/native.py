"""Literal-only P1 null/terminal native endpoint; inert without --run.

Reuse the qualified native loop and passive observer. Own complete bills,
domain integrity and exact repeats are mandatory. Changed terminal P may
change all layers and selector widths; no old-cost transplantation occurs.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
TL = NEW/'current_d50_terminal_linear_v2'
BD = NEW/'birth_direction_fit_v1'
PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
BUILD_SOURCE_SHA = '57f87081cdcc742266bc8487ef5ba11358bdf25e8341b9b1f4dd4458f06da54c'
WORKER_SHA = 'd1160bfb2f5ac0edf4d425f09282ec79bbca310cb2f88cf7b97b6c59acbd426c'
COMMON_SHA = '567ccbae6caf516043d72d54a9073cc60e7fafeac31934d29379ef578fea0a7f'
WATCH_SHA = '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3'
TL_NATIVE_SHA = '69c8eeee8078feb2b12662f128ea873e5d67435182d4c30bd551a1369c84f69c'
CAPTURE_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
TL_COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
SCHEMA = 'current_d50_native_objective_p1_native_v1'
BUDGET, INITIALIZATION = 2**41, 1383913786
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No target-array admission')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def legacy():
    require(sha(TL/'native_validation.py') == TL_NATIVE_SHA, 'Qualified native adapter')
    return module(TL/'native_validation.py','p1_native_legacy_admission')


def admission(build_sha256, source_sha256, pin):
    for path,digest in ((HERE/'native.py',source_sha256),(HERE/'build.py',BUILD_SOURCE_SHA),
        (HERE/'worker.py',WORKER_SHA),(HERE/'common.py',COMMON_SHA),(HERE/'watch.py',WATCH_SHA),
        (HERE/'PROTOCOL.md',PROTOCOL_SHA),(TL/'native_validation.py',TL_NATIVE_SHA),
        (TL/'native.py',CAPTURE_SHA),(TL/'common.py',TL_COMMON_SHA),(BD/'native_v2.py',NATIVE_SHA)):
        pin(path,digest)
    tl = legacy()
    builder = module(HERE/'build.py','p1_native_terminal_builder_metadata')
    build = builder.sealed(HERE/'build_v1',build_sha256)
    require(build['schema'] == 'current_d50_native_objective_p1_v1' and build['stage'] == 'build'
            and build['terminal_epoch'] == 1 and build['updates'] == 40
            and build['parent_sha256'] == build['null_sha256'] == tl.BANK_SHA
            and build['modified_assignment_only'] == '_GNN_LITERAL'
            and all(build[k] is True for k in ('inverse_source_byte_exact','null_source_byte_exact',
                'non_PKEYS_byte_exact','unused_layer0_byte_exact','cstar_byte_exact',
                'terminal_literal_roundtrip_byte_exact')),
            'Sole completed P-literal proposal, no changed card arithmetic')
    helper,training,keys,sources,inputs = builder.admission(SimpleNamespace(
        source_sha256=BUILD_SOURCE_SHA,worker_sha256=WORKER_SHA,common_sha256=COMMON_SHA,
        watcher_sha256=WATCH_SHA,training_sha256=build['training_result_sha256'],
        training_watch_sha256=build['training_watch_sha256']))
    require(build['terminal_payload'] == training['terminal_payload'], 'Built exact terminal payload')
    for bindings in (sources,inputs,build['source_bindings'],build['input_bindings'],build['artifacts']):
        tl.bind(bindings,pin)
    pin(HERE/'build_v1/result.json',build_sha256); pin(HERE/'build_v1/completion.json')
    for group in ('source_bindings','input_bindings','weight_bindings','artifacts'):
        tl.bind(training[group],pin)
    parent = tl.BANK.read_bytes()
    candidate, null = HERE/'build_v1/estimator.py', HERE/'build_v1/null_estimator.py'
    pin(candidate,build['candidate_sha256']); pin(null,tl.BANK_SHA)
    require(null.read_bytes() == parent, 'Null source BYTE exact to current bank')
    a,b,p = helper.assignment_span(parent)
    terminal = candidate.read_bytes(); x,y,q = helper.assignment_span(terminal)
    require(terminal[:x]+parent[a:b]+terminal[y:] == parent and set(p) == set(q), 'P-only source inverse')
    for key in p:
        if key not in keys:
            require(tl.f32(p[key]) == tl.f32(q[key]), 'Frozen non-P bytes')
    for key in ('head','head_b','gain'):
        require(tl.f32(p[key][0]) == tl.f32(q[key][0]), 'Frozen inactive layer0')
    control,control_outputs = tl.control(pin)
    return dict(build=build,training=training,build_result_sha256=build_sha256,
        candidate=candidate,candidate_sha256=build['candidate_sha256'],null=null,
        control=control,control_outputs=control_outputs,control_result_sha256=tl.CONTROL_SHA)


def native_inputs(candidate, args):
    require(args.protocol_sha256 == PROTOCOL_SHA and sha(__file__) == args.source_sha256,
            'Reviewed native and prospective protocol')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'Admission before numeric imports')
    tl = legacy()
    require({k:importlib.metadata.version(k) for k in tl.VERSIONS} == tl.VERSIONS, 'Qualified OFFPY versions')
    require(sys.flags.optimize == 0 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and all(os.environ.get(k) == '1' for k in tl.BLAS)
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Assertions and fixed BLAS/fallback environment')
    inputs = {}; pin = tl.pinner(inputs)
    context = admission(args.build_sha256,args.source_sha256,pin)
    require(candidate == context['null' if args.role == 'null' else 'candidate'], 'Exact selected role source')
    context['role'] = args.role
    context['ranks'] = (VALID[0],) if args.role == 'null' else VALID
    return inputs,context


def check_call(result, trial, values, packet, helper, np, context):
    call = trial['call']
    old_call = (0 if call == 0 else 40) if context['role'] == 'null' else call
    old = context['control']['trials'][old_call]
    require(trial['rank'] == old['rank'] and trial['includes_initialization'] == old['includes_initialization'],
            'Same current fixture and complete initialization ownership')
    if context['role'] == 'null':
        require(values.tobytes() == context['control_outputs'][old_call]
                and trial['flops'] == old['flops'] and trial['operation_count'] == old['operation_count'],
                'Null all16 BYTE and own complete F/opcount parity')
    domain = helper.diagnostics(np,packet)
    domain['selection_cells'] = {key:{k:item[k] for k in
        ('count','floor','indices_sha256','mask_sha256')} for key,item in domain['selection_cells'].items()}
    result['domain_diagnostics'].append(dict(call=call,rank=trial['rank'],**domain))


def finish_native(result, context):
    ranks = list(context['ranks'])+[context['ranks'][0]]
    count = len(ranks)
    require(len(result['trials']) == len(result['domain_diagnostics']) == count
            and [x['rank'] for x in result['trials']] == ranks, 'Exact native roster plus repeat')
    require(len(result['artifacts']) == 5*count and len(result['input_preparation']) == count, 'Original lifecycles complete')
    initialization = result['trials'][0]['flops']-result['trials'][-1]['flops']
    require(initialization == INITIALIZATION, 'Same-shaped initialization actually charged once')
    require(result['trials'][0]['output_bytes_sha256'] == result['trials'][-1]['output_bytes_sha256'], 'Repeat output BYTE')
    require({k:v for k,v in result['domain_diagnostics'][0].items() if k != 'call'} ==
            {k:v for k,v in result['domain_diagnostics'][-1].items() if k != 'call'}, 'Repeat native chart BYTE')
    result.update(passed=True,completed=True,hashes_unchanged=True,initialization_flops=initialization,
        all16_domains_qualified=True,all47_selectors_qualified=True,own_complete_bills=True,
        all_outputs_and_bills_sealed_before_scoring=True,score_measured=False,
        null_full_output_bill_parity=context['role'] == 'null')


def derived_main():
    tl = legacy()
    source,original_sha = tl.derived_main()
    edits = [
        ("for name in ('source','protocol','fit')", "for name in ('source','protocol','build')"),
        ('    args=parser.parse_args()\n', "    parser.add_argument('--role',choices=('null','terminal'),required=True)\n    args=parser.parse_args()\n"),
        ("    args.source=HERE/'fit_v1/estimator.py'\n",
         "    args.source=HERE/'build_v1'/('null_estimator.py' if args.role=='null' else 'estimator.py')\n"),
        ("args.out=HERE/'validation_terminal_v1'", "args.out=HERE/f'native_{args.role}_v1'"),
        ('    ranks=VALID\n', '    ranks=prerequisite["ranks"]\n'),
        ("schema='current_d50_terminal_linear_native40_v1'", "schema=SCHEMA,stage='native',role=args.role"),
        ('        prerequisites=dict(fit_result_sha256=args.fit_sha256,fit_watch_sha256=prerequisite["fit_watch_sha256"],\n'
         '            source_inverse=prerequisite["proof"],control_result_sha256=CONTROL_SHA),',
         '        prerequisites=dict(build_result_sha256=args.build_sha256,control_result_sha256=CONTROL_SHA,\n'
         '            training_result_sha256=prerequisite["build"]["training_result_sha256"],\n'
         '            training_watch_sha256=prerequisite["build"]["training_watch_sha256"]),'),
    ]
    changed = source
    for old,new in edits:
        require(changed.count(old) == 1, 'Unique P1 admission/role metadata splice')
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique P1 adapter inverse')
        inverse = inverse.replace(new,old)
    require(inverse == source, 'Exact inverse to qualified passive native loop')
    return changed,original_sha


def main():
    if '--run' not in sys.argv:
        print('Inert: --run --role null|terminal --source-sha256 SHA --protocol-sha256 SHA --build-sha256 SHA')
        return 0
    os.chdir(NEW)
    tl = legacy(); source,digest = derived_main()
    require(sha(TL/'native.py') == CAPTURE_SHA and sha(TL/'common.py') == TL_COMMON_SHA, 'Original observer sources')
    previous = sys.modules.get('common')
    sys.modules['common'] = module(TL/'common.py','p1_passive_observer_common')
    try:
        capture = module(TL/'native.py','p1_passive_native_observer')
    finally:
        if previous is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common'] = previous
    base = module(BD/'native_v2.py','p1_unchanged_native_loop')
    env = dict(vars(base),__file__=str(Path(__file__).resolve()),HERE=HERE,PARENT=NEW,
        native_inputs=native_inputs,observed_predict=tl.observed_predict,check_call=check_call,
        finish_native=finish_native,CAPTURE_HELPER=capture,MAIN_SHA=digest,SCHEMA=SCHEMA,
        NATIVE_SHA=NATIVE_SHA,CONTROL_SHA=tl.CONTROL_SHA,VERSIONS=tl.VERSIONS)
    exec(compile(source,str(__file__)+':preserved_native_main','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
