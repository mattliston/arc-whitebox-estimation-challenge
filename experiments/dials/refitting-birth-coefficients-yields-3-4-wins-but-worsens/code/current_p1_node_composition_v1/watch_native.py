"""One fixed zero or terminal native phase; inert without --run.

Exact-inverse adapter to the qualified census supervisor. The owned-group
cleanup, independent sampling and next completed-bill/lifecycle deadline
are preserved. No numerical imports, targets, automatic retry or scoring.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BASE = NEW/'current_d50_response10_census_v1/watch.py'
BASE_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
PROTOCOL_SHA = '5aa1fefd2e7b60ce77814f39efc4dcea3bcd2cc7ed372d4841da4eb8f973696c'
SCHEMA = 'current_p1_node_composition_native_watch_v1'
CHILD_SCHEMA = 'current_p1_node_composition_native_v1'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
FIXED = {'zero':(35,100),'terminal':VALID}


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'Never open target arrays')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            result.update(block)
    return result.hexdigest()


def load(path,name,expected):
    require(re.fullmatch('[0-9a-f]{64}',expected or '') and sha(path) == expected,
            'Reviewed stdlib-only owner: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def admit(args,worker,helper,helper_sha):
    """The numerical worker's replayable stdlib source/receipt admission."""
    native = load(worker,'node_p1_native_admission',args.worker_sha256)
    require(native.FIXED == FIXED and native.SCHEMA == CHILD_SCHEMA and native.WATCH_SCHEMA == SCHEMA,
            'Same role/schema contract')
    context = native.authenticate(args.role,args.build_sha256,args.candidate_sha256,
        args.builder_sha256,args.worker_sha256,args.source_sha256,
        args.zero_result_sha256,args.zero_watch_sha256)
    sources,inputs = dict(context['source_bindings']),dict(context['input_bindings'])
    sources.update({str(BASE):BASE_SHA,str(helper):helper_sha})
    native.bind(sources); native.bind(inputs)
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'No numerical imports during admission')
    return sources,inputs,dict(context,native_module=native)


def validate_child(record,end,digest,args,folder,control):
    require(folder == HERE/f'native_{args.role}_v1' and record['prerequisites'] == control['prerequisites'],
            'Exact stage and zero-admission lineage')
    control['native_module'].verify_acquisition(record,end,digest,args.role,
        args.worker_sha256,args.source_sha256,control['reference_by_rank'])


def derived_main(base):
    text = BASE.read_text()
    node = next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    start = original.index('    sources = {str(Path(__file__).resolve()):args.source_sha256')
    stop = original.index('    require(not HERE.is_symlink()',start)
    edits = [
        ("    parser.add_argument('--run',action='store_true')\n",
         "    parser.add_argument('--run',action='store_true')\n"
         "    parser.add_argument('--role',choices=tuple(FIXED))\n"
         "    parser.add_argument('--kernel-verdict-reviewed',action='store_true')\n"
         "    parser.add_argument('--numerical-slot-clear',action='store_true')\n"),
        ("for name in ('source','worker','protocol'):",
         "for name in ('source','worker','protocol','builder','build','candidate','zero-result','zero-watch'):"),
        ("for k in ('source','worker','protocol'))",
         "for k in ('source','worker','protocol','builder','build','candidate'))"),
        ('    began = time.monotonic()\n',
         '    began = time.monotonic()\n'
         '    require(args.role in FIXED and args.kernel_verdict_reviewed and args.numerical_slot_clear,\n'
         "            'Explicit reviewed prior-kernel verdict and free numerical slot')\n"
         '    sequence = FIXED[args.role]+(FIXED[args.role][0],)\n'
         "    WALL_LIMIT = 240.0 if args.role == 'zero' else 1900.0\n"),
        ("'Sole fixed response.1 census protocol'", "'Sole frozen NODE/P1 compatibility protocol'"),
        ("worker,child_out,out = HERE/'capture.py',HERE/'native',HERE/'native_watch'",
         "worker,child_out,out = HERE/'native.py',HERE/f'native_{args.role}_v1',HERE/f'native_{args.role}_watch_v1'"),
        (original[start:stop], '    sources,inputs,control = admit(args,worker,HELPER,HELPER_SHA)\n'),
        ("'--source-sha256',args.worker_sha256,'--protocol-sha256',PROTOCOL_SHA]",
         "'--source-sha256',args.worker_sha256,'--protocol-sha256',PROTOCOL_SHA,\n"
         "        '--role',args.role,'--builder-sha256',args.builder_sha256,'--build-sha256',args.build_sha256,\n"
         "        '--candidate-sha256',args.candidate_sha256,'--watcher-sha256',args.source_sha256]\n"
         "    if args.role == 'terminal':\n"
         "        command += ['--zero-result-sha256',args.zero_result_sha256,'--zero-watch-sha256',args.zero_watch_sha256]"),
        ('candidate_sha256=BANK_SHA,control_result_sha256=CONTROL_SHA,truth_arrays_opened=False,',
         'role=args.role,stage=\'native\',candidate_sha256=args.candidate_sha256,build_sha256=args.build_sha256,\n'
         '        prerequisites=control[\'prerequisites\'],truth_arrays_opened=False,truth_accessed=False,\n'
         '        kernel_verdict_reviewed=True,numerical_slot_clear=True,kernel_pass_required=False,\n'
         '        auto_launch_next_stage=False,old_NODE_gate_still_failed=True,'),
        ("len(found) <= 2,'Ordered call00/01 only'", "len(found) <= len(sequence),'Only the fixed ordered role roster'"),
        ("bill['rank'] == 35 and", "bill['rank'] == sequence[call] and"),
        ('dict(call=call,rank=35,path=str(bp)', 'dict(call=call,rank=sequence[call],path=str(bp)'),
        ("result['completed_bill_count'] == 2,'Both completed lifecycles observed in order'",
         "result['completed_bill_count'] == len(sequence),'Every fixed bill/lifecycle observed in order'"),
        ("        started_utc=datetime.now(timezone.utc).isoformat())",
         "        watch_source_inverse_verified=True,watch_parent_sha256=BASE_SHA,\n"
         "        started_utc=datetime.now(timezone.utc).isoformat())")]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1,'Unique watchdog splice: '+before[:80])
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique watchdog inverse')
        inverse = inverse.replace(after,before)
    require(inverse == original,'All other resource/progress/deadline/cleanup statements unchanged')
    return changed


def main():
    base = load(BASE,'node_p1_watch_parent',BASE_SHA)
    source = derived_main(base)
    namespace = dict(vars(base),__file__=str(Path(__file__).resolve()),__doc__=__doc__,HERE=HERE,
        SCHEMA=SCHEMA,CHILD_SCHEMA=CHILD_SCHEMA,PROTOCOL_SHA=PROTOCOL_SHA,BASE_SHA=BASE_SHA,
        SWAP_LIMIT=0,FIXED=FIXED,admit=admit,validate_child=validate_child)
    exec(compile(source,str(__file__)+':preserved_watch','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
