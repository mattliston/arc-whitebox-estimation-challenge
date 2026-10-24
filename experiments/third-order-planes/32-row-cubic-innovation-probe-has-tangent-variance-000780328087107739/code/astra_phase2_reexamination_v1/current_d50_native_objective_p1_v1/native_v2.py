"""Additive matched-history null repair; the trained estimator is unchanged."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD_SHA = 'ebab249a8a1bdd0c2017a1a24736c98eac708310865f50588d7ab4043170e2d4'
ENDPOINT_SHA = '57d0f5e464d0c2f4313336426c0c5cde0ab0d95442305c00ad5c5b9c2a7e4021'
FAILED_SHA = '019f3f29df183e337b862c2bb6e540494a8cc5245cf5d51359d091d8b4d5406e'
FAILED_WATCH_SHA = '46082a7bc015e496291cfc173c608f421c44414721d748079ba0d76ef1eea231'
SCHEMA = 'current_d50_native_objective_p1_native_v2'


def old():
    import hashlib
    path = HERE/'native.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == OLD_SHA
    spec = importlib.util.spec_from_file_location('p1_native_v1_preserved',path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def legacy():
    return old().legacy()


NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
INITIALIZATION = 1383913786


def admission(build_sha256,source_sha256,pin):
    n = old()
    pin(__file__,source_sha256); pin(HERE/'ENDPOINT_V2.md',ENDPOINT_SHA)
    pin(HERE/'native_null_v1/result.json',FAILED_SHA)
    pin(HERE/'native_null_watch_v1/result.json',FAILED_WATCH_SHA)
    for folder,digest in ((HERE/'native_null_v1',FAILED_SHA),(HERE/'native_null_watch_v1',FAILED_WATCH_SHA)):
        pin(folder/'completion.json')
        record,end = n.read(folder/'result.json'),n.read(folder/'completion.json')
        n.require(record['passed'] is False and record['completed'] is False and
                  end['passed'] is False and end['completed'] is False and end['result_sha256'] == digest,
                  'Preserved failed endpoint1 receipts remain failed')
    return n.admission(build_sha256,OLD_SHA,pin)


def native_inputs(candidate,args):
    n = old(); tl = n.legacy()
    n.require(n.sha(__file__) == args.source_sha256 and args.protocol_sha256 == PROTOCOL_SHA,
              'Reviewed endpoint2 source and fixed P1 protocol')
    n.require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'Admission before numerical imports')
    n.require({k:n.importlib.metadata.version(k) for k in tl.VERSIONS} == tl.VERSIONS,
              'Same qualified OFFPY libraries')
    n.require(sys.flags.optimize == 0 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
              and all(os.environ.get(k) == '1' for k in tl.BLAS)
              and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0','Same fixed native launch frame')
    pins = {}; pin = tl.pinner(pins)
    context = admission(args.build_sha256,args.source_sha256,pin)
    expected = tl.BANK if args.role == 'bank' else context['null' if args.role == 'null' else 'candidate']
    n.require(candidate == expected,'Exactly the direct bank, built null or sole terminal')
    context.update(role=args.role,ranks=VALID if args.role == 'terminal' else (VALID[0],))
    if args.role == 'null':
        folder = HERE/'native2_bank_v1'; digest = pin(folder/'result.json')
        record = tl.sealed(folder,digest,pin)
        n.require(record['schema'] == SCHEMA and record['stage'] == 'native' and record['role'] == 'bank'
                  and record['source_bindings'][str(HERE/'native_v2.py')] == args.source_sha256
                  and record['prerequisites']['build_result_sha256'] == args.build_sha256,
                  'Fresh direct-bank control has the same source/build/history')
        _,verify = tl.verifier(tl.BANK,tl.BANK_SHA,pin)
        context.update(control=record,control_outputs=verify(record,folder,(VALID[0],),pin),
                       bank_result_sha256=digest)
    return pins,context


def check_call(result,trial,values,packet,helper,np,context):
    n = old(); call = trial['call']; role = context['role']
    control_call = (0 if call == 0 else 40) if role == 'bank' else call
    control = context['control']['trials'][control_call]
    n.require(trial['rank'] == control['rank'] and
              trial['includes_initialization'] == control['includes_initialization'],'Matching input/cold ownership')
    if role in ('bank','null'):
        n.require(values.tobytes() == context['control_outputs'][control_call] and
                  trial['flops'] == control['flops'],'Actual all16 bytes and complete FLOPs match')
        if role == 'null' or call == 0:
            n.require(trial['operation_count'] == control['operation_count'],'Exact operation count on matching history')
    domain = helper.diagnostics(np,packet)
    domain['selection_cells'] = {key:{k:item[k] for k in
        ('count','floor','indices_sha256','mask_sha256')} for key,item in domain['selection_cells'].items()}
    result['domain_diagnostics'].append(dict(call=call,rank=trial['rank'],**domain))


def derived_main():
    n = old(); source,digest = n.derived_main()
    edits = [
        ("choices=('null','terminal')","choices=('bank','null','terminal')"),
        ("    args.source=HERE/'build_v1'/('null_estimator.py' if args.role=='null' else 'estimator.py')\n",
         "    args.source=BANK_SOURCE if args.role=='bank' else HERE/'build_v1'/('null_estimator.py' if args.role=='null' else 'estimator.py')\n"),
        ("args.out=HERE/f'native_{args.role}_v1'","args.out=HERE/f'native2_{args.role}_v1'"),
        ('        prerequisites=dict(build_result_sha256=args.build_sha256,control_result_sha256=CONTROL_SHA,\n',
         '        prerequisites=dict(build_result_sha256=args.build_sha256,control_result_sha256=CONTROL_SHA,\n'
         '            bank_result_sha256=prerequisite.get("bank_result_sha256"),\n'),
    ]
    changed = source
    for before,after in edits:
        n.require(changed.count(before) == 1,'Unique matched-fixture metadata splice')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        n.require(inverse.count(after) == 1,'Unique matched-fixture inverse')
        inverse = inverse.replace(after,before)
    n.require(inverse == source,'Every remaining native statement unchanged')
    return changed,digest


def main():
    if '--run' not in sys.argv:
        print('Inert matched-history native endpoint2; use --run with reviewed hashes and role.')
        return 0
    n = old(); tl = n.legacy(); source,digest = derived_main()
    os.chdir(n.NEW)
    n.require(n.sha(HERE/'ENDPOINT_V2.md') == ENDPOINT_SHA and
              n.sha(n.TL/'native.py') == n.CAPTURE_SHA and n.sha(n.TL/'common.py') == n.TL_COMMON_SHA,
              'Unchanged protocol and passive observer')
    previous = sys.modules.get('common')
    sys.modules['common'] = n.module(n.TL/'common.py','p1_native2_passive_common')
    try:
        capture = n.module(n.TL/'native.py','p1_native2_passive_observer')
    finally:
        if previous is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common'] = previous
    base = n.module(n.BD/'native_v2.py','p1_native2_unchanged_loop')
    env = dict(vars(base),__file__=str(Path(__file__).resolve()),HERE=HERE,PARENT=n.NEW,
        native_inputs=native_inputs,observed_predict=tl.observed_predict,check_call=check_call,
        finish_native=n.finish_native,CAPTURE_HELPER=capture,MAIN_SHA=digest,SCHEMA=SCHEMA,
        BANK_SOURCE=tl.BANK,NATIVE_SHA=NATIVE_SHA,CONTROL_SHA=tl.CONTROL_SHA,VERSIONS=tl.VERSIONS)
    exec(compile(source,str(__file__)+':preserved_native_main','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
