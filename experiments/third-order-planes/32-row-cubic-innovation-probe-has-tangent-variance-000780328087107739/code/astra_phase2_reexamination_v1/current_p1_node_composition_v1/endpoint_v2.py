"""Lightweight stdlib-only owners and sealed-context checks; never constructs P."""
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

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
ENDPOINT_SHA = '9ff7b778bd04069b693e57d3005c1d9755bf3ee755d1e3f2be6744c6a8948794'
OLD_NATIVE_SHA = 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01'
OLD_WATCH_SHA = '7cf461c4c5e847a81c8b20b6c6c37f7f9169aee1d0d4507382e2842caed8d53b'
FAIL_SHA = '56046ed7ef0ea50c7af5e5e8f5cbe51a38181721bd753b38f5070477eda0f66d'
FAIL_END_SHA = 'd969deadafeee51107763829eed1a07adeef75b7867d7787095ff2a0169c0ceb'
ZERO_SHA = '459122d34f41759700e0cdff43c799072afe1be9853eccee3dc4e83e36680fc4'
ZERO_WATCH_SHA = '8fd4a858bb281ec665c0ec32b179f17f67e798da51aec414e3498f9e05635a66'
HELPER = NEW/'current_d50_terminal_linear_v2/watch_fit.py'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
ADMISSION = HERE/'admission_v2'
OUTPUT = HERE/'native_terminal_v2'
WATCH_OUT = HERE/'native_terminal_watch_v2'
CONTEXT_SCHEMA = 'current_p1_node_composition_context_v2'
ADMIT_SCHEMA = 'current_p1_node_composition_admission_v2'
NATIVE_SCHEMA = 'current_p1_node_composition_native_v2'
WATCH_SCHEMA = 'current_p1_node_composition_native_watch_v2'
CONTEXT_LIMIT = 2*1024**2
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
OWNER_FILES = {'watcher':'watch_native_v2.py','worker':'native_v2.py','admitter':'admit_v2.py',
               'helper':'endpoint_v2.py','endpoint':'ENDPOINT_MEMORY_V2.md'}
DIAGNOSTIC_SHAS = (
 '064e6919251896826c3fc2658b4f9ddb280eebcd264827ddb7053a73646d7354',
 '2e08777dc37689c06041429187893338f0be41b9b848e8a5b0adffe53254021f',
 'c5f93f9300514528246b806eb3cf98e94d47a07a91dcfa14a08e1681035587af',
 'a46fab1e480805669d54a1fbd0cf449f657a372f38d458ff83c9ad65dd0bf5b7',
 'f4cb7eb66ba188416b0ac8e03447492b4696f6e6edce7e03f72c4894f2ce7c83',
 'f6545a7beabb5ca0db14188832cad643e6d68244fc295e321e697726a71bef27',
 '29e2510a6a2dbc9d04a17794ac3e3015202e23881598a39c6084daa47c19c1d7',
 'f5d2cdcd2125054a971c490b8a3088543598bf41a5b6244c625c6c778b0a5f48',
 '22b171f5519dab9657be94e34407f11088a73fa9a832f585135fee1f07bf3501',
 'b468752c867a18ecad0425972d1781166e98ba21f1310a154a69c42951118fbc',
 'c4dbd3a1cb6257e5d8028fdb88b94fdabb2287d0f202c145d30cced85d984852',
 'e4377a3a9eccb4ca63fb4a00da2e657ff239fbfd1f820f641a51a83b2dfe9549',
 '63455462ae83df594d8c3e0fd53103f9d7a71c8f3c0666d49be5db0edb4b9c20',
 '7cdeebe65ed64f0cbb970baceb21329be09f4a4aa2bfb966818e6e56b3efd8e9',
 '00a73aa7d6a39760ca0f551419f538e3d5ca811f017e512b4ce3044799d128dc',
 'f71619d55af1e6330d139e2a26ee87b5aa3eb7c86955b2cfa70a1737b5ca70c9',
 '41e19dfbc8a62ce0e6d0d5a44fd68b7952a92bbf18d30527cff3f81625782c97',
 'ce8119be950c0dd227a233c5a85b572a76eb0324aa30807c320d4bae45f99a16',
 '20fb813d4e2764dca11be32b497d0a3ae8ab851504e62d61d98c469eb6049b9e',
 'fa9155965c4562c73d29aa283f9da2867fade36ef840ddb7cd6122bba104d55a',
 '0208c6e2a1502aac4bd20bbe771a06b7bc85de4ec15190bd8fa403885f5f2914',
 '61d366498a937dd04d209546e4239a2bbd07223cf1702a3781d261193a0478ca')
_OLD = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'Target arrays forbidden')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick: tick()
    return digest.hexdigest()


def read(path, bound=None):
    path = Path(path)
    require(not path.is_symlink() and (bound is None or path.stat().st_size <= bound), 'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def bind(items, tick=None):
    require(isinstance(items,dict), 'Flat hash map')
    for path,digest in items.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or '')
                and sha(path,tick) == digest, 'Immutable dependency: '+str(path))


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed source owner')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old():
    global _OLD
    if _OLD is None:
        _OLD = load(HERE/'native.py','node_p1_frozen_v1_metadata',OLD_NATIVE_SHA)
    return _OLD


def arguments(role):
    parser = argparse.ArgumentParser(description='Fixed NODE/P1 endpoint-memory v2: '+role)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--watcher-sha256','--source-sha256',dest='watcher_sha256')
    for name in ('worker','admitter','helper','endpoint','context','admission-watch'):
        parser.add_argument('--'+name+'-sha256')
    if role == 'watcher':
        parser.add_argument('--prior-verdict-reviewed',action='store_true')
        parser.add_argument('--numerical-slot-clear',action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return None
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in OWNER_FILES), 'All reviewed owners')
    require(args.endpoint_sha256 == ENDPOINT_SHA and (args.context_sha256 is not None) == (role == 'worker')
            and (args.admission_watch_sha256 is not None) == (role == 'worker'),
            'Fixed supplement and correct context stage')
    if role == 'worker':
        require(all(re.fullmatch('[0-9a-f]{64}',x) for x in (args.context_sha256,args.admission_watch_sha256)),
                'Parent-supplied context and retired-admission seals')
    return args


def owners(args, tick=None):
    result = {str(HERE/name):getattr(args,key+'_sha256') for key,name in OWNER_FILES.items()}
    result.update({str(HERE/'native.py'):OLD_NATIVE_SHA,str(HERE/'watch_native.py'):OLD_WATCH_SHA,
                   str(HELPER):HELPER_SHA})
    bind(result,tick)
    return result


def command(args, role, context_sha=None, admission_watch_sha=None):
    cmd = [str(OFFPY),'-B','-u',str(HERE/OWNER_FILES[role]),'--run']
    for key in OWNER_FILES:
        cmd += ['--'+key+'-sha256',getattr(args,key+'_sha256')]
    if context_sha: cmd += ['--context-sha256',context_sha]
    if admission_watch_sha: cmd += ['--admission-watch-sha256',admission_watch_sha]
    return cmd


def environment():
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
            and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'OFFPY/nice10/BLAS1/no fallback')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')), 'Pre-numerical stdlib stage')


def function_source(path, name):
    text = Path(path).read_text()
    nodes = [x for x in ast.parse(text).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name == name]
    require(len(nodes) == 1, 'Unique frozen source object')
    node = nodes[0]
    return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def splice(source, edits):
    result = source
    for before,after in edits:
        require(result.count(before) == 1, 'Unique source substitution'); result = result.replace(before,after)
    restored = result
    for before,after in reversed(edits):
        require(restored.count(after) == 1, 'Unique source inverse'); restored = restored.replace(after,before)
    require(restored == source, 'Full source inverse')
    return result


def preserved22(pins):
    """Bind completed prefix of a FAILED run; no success/score inference."""
    n = old(); folder = HERE/'native_terminal_v1'; watch = HERE/'native_terminal_watch_v1'
    pins.update({str(watch/'result.json'):FAIL_SHA,str(watch/'completion.json'):FAIL_END_SHA}); bind(pins)
    failed,end = read(watch/'result.json'),read(watch/'completion.json')
    require(failed['passed'] is False and failed['completed'] is False and end['passed'] is False
            and end['result_sha256'] == FAIL_SHA and failed['whole_owned_tree_retired']
            and failed['completed_bill_count'] == len(failed['progress']) == 22
            and failed['maximum_sampled_rss_bytes'] == 8020918272
            and failed['source_bindings'][str(HERE/'native.py')] == OLD_NATIVE_SHA
            and failed['source_bindings'][str(HERE/'watch_native.py')] == OLD_WATCH_SHA, 'Preserved memory failure')
    for group in ('source_bindings','input_bindings','artifacts'):
        bind(failed[group]); pins.update(failed[group])
    helper = load(n.CENSUS,'node_p1_old_prefix_lifecycle',n.CENSUS_SHA)
    records = []
    for call,(item,diagnostic_sha) in enumerate(zip(failed['progress'],DIAGNOSTIC_SHAS)):
        rank = n.VALID[call]
        bp,lp,dp = folder/f'call_{call:02d}_bill.json',folder/f'call_{call:02d}_lifecycle.json',folder/f'call_{call:02d}_diagnostics.json'
        require(item['call'] == call and item['rank'] == rank and item['path'] == str(bp)
                and item['lifecycle_path'] == str(lp), 'Exact completed old prefix order')
        local = {str(bp):item['sha256'],str(lp):item['lifecycle_sha256'],str(dp):diagnostic_sha}; bind(local)
        bill = read(bp); helper.lifecycle(bill,bp,lp)
        ip,ir,op = folder/f'call_{call:02d}_input_bill.json',folder/f'call_{call:02d}_input_reset.json',folder/f'call_{call:02d}_rank{rank}.npy'
        require(bill['call'] == call and bill['rank'] == rank and bill['input_bill'] == str(ip)
                and bill['input_reset_receipt'] == str(ir) and bill['output'] == str(op), 'Old owned artifacts')
        local.update({str(ip):bill['input_bill_sha256'],str(ir):bill['input_reset_sha256'],str(op):bill['output_sha256']})
        bind(local); helper.lifecycle(dict(read(ip),lifecycle_receipt=str(ir)),ip,ir)
        n.npy_bytes(op,bill['output_bytes_sha256'])
        diagnostic = read(dp)
        require(diagnostic['call'] == call and diagnostic['rank'] == rank and diagnostic['truth_accessed'] is False
                and diagnostic['offline_after_saved_bill_and_context_release'], 'Old detached diagnostics')
        pins.update(local); records.append(dict(call=call,rank=rank,bill=str(bp),input_bill=str(ip),
            diagnostics=str(dp),diagnostics_sha256=diagnostic_sha,bindings=local))
    return records


def compare22(trial, prepared, diagnostic_path, context):
    call = trial['call']
    if call >= 22: return False
    item = context['preserved22'][call]; bind(item['bindings'])
    before,old_input = read(item['bill']),read(item['input_bill'])
    for key in ('call','rank','flops','operation_count','multiplier','includes_initialization','output_sha256','output_bytes_sha256'):
        require(trial[key] == before[key], 'Same-history old22 exact output/bill field: '+key)
    timing = {'wall_s','context_wall_s','flopscope_backend_s','flopscope_overhead_s','native_residual_s','peak_rss_bytes'}
    require(set(prepared) == set(old_input) and all(prepared[k] == old_input[k] for k in old_input if k not in timing),
            'Same-history old22 exact official-input bill/ownership')
    require(sha(trial['output']) == before['output_sha256'] and sha(diagnostic_path) == item['diagnostics_sha256'],
            'Same-history all16 output and full startup/domain/selector JSON bytes')
    return True


def read_context(args, digest, tick=None):
    expected = owners(args,tick)
    rp,cp,jp = ADMISSION/'result.json',ADMISSION/'completion.json',ADMISSION/'context.json'
    require(sha(jp,tick) == digest, 'Exact parent-supplied bounded context')
    value = read(jp,CONTEXT_LIMIT); record,end = read(rp,CONTEXT_LIMIT),read(cp,CONTEXT_LIMIT)
    require(value['schema'] == CONTEXT_SCHEMA and record['schema'] == ADMIT_SCHEMA
            and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == sha(rp,tick)
            and record['context_sha256'] == digest and value['reviewed_owners'] == expected
            and record['reviewed_owners'] == expected and record['estimator_executed'] is False
            and record['numerical_imports'] is False and record['truth_accessed'] is False,
            'Reviewed admission producer and exact completed context seal')
    n = old()
    require(value['candidate_sha256'] == n.CANDIDATES['terminal'] and value['build_sha256'] == n.BUILD_SHA
            and value['role'] == 'terminal' and value['ranks'] == list(n.VALID)
            and value['prerequisites'] == dict(build_sha256=n.BUILD_SHA,zero_result_sha256=ZERO_SHA,zero_watch_sha256=ZERO_WATCH_SHA)
            and len(value['preserved22']) == 22 and value['failed_watch_sha256'] == FAIL_SHA
            and value['full_frozen_admission_passed'] and value['observer_inverse_verified']
            and value['paid_loop_inverse_verified'], 'Fixed complete restart and preserved original proofs')
    for group in ('source_bindings','input_bindings'):
        bind(value[group],tick)
    require(all(value['source_bindings'].get(k) == v for k,v in expected.items()), 'Every reviewed owner bound')
    value['input_bindings'].update({str(jp):digest,str(rp):sha(rp,tick),str(cp):sha(cp,tick)})
    value['reference_by_rank'] = {int(k):v for k,v in value['reference_by_rank'].items()}
    return value


def retired_admission(args, context):
    path = WATCH_OUT/'admission_retired.json'
    require(sha(path) == args.admission_watch_sha256, 'Exact independently retired admission seal')
    value = read(path,CONTEXT_LIMIT)
    require(value['schema'] == 'current_p1_node_composition_admission_watch_v2'
            and value['passed'] and value['completed'] and value['whole_owned_tree_retired']
            and value['context_sha256'] == args.context_sha256 and value['reviewed_owners'] == owners(args)
            and value['admission_elapsed_s'] < 60 and value['whole_elapsed_s'] < 120
            and value['maximum_sampled_rss_bytes'] < 8_000_000_000
            and value['maximum_swap_growth_bytes'] == 0 and value['maximum_sample_interval_s'] <= 1,
            'The owned admission process retired under the unchanged resource limits')
    context['input_bindings'][str(path)] = args.admission_watch_sha256
    return value


def verify_native(record,end,digest,args,context):
    n = old()
    source = function_source(HERE/'native.py','verify_acquisition')
    derived = splice(source,[("f'native_{role}_v1'","f'native_{role}_v2'"),
        ("HERE/'native.py'","HERE/'native_v2.py'"),("HERE/'watch_native.py'","HERE/'watch_native_v2.py'")])
    namespace = dict(vars(n),SCHEMA=NATIVE_SCHEMA)
    exec(compile(derived,str(__file__)+':unchanged_verifier','exec'),namespace)
    namespace['verify_acquisition'](record,end,digest,'terminal',args.worker_sha256,args.watcher_sha256,context['reference_by_rank'])
    require(record['admission_context_sha256'] == args.context_sha256
            and record['admission_watch_sha256'] == args.admission_watch_sha256 and record['first22_exact_checks'] == 22
            and record['first22_output_bill_domain_byte_exact'] and record['observer_event_filter_inverse_verified']
            and record['failed_v1_watch_sha256'] == FAIL_SHA, 'Versioned acquisition and strengthened22-call replay')
    for call in range(22):
        compare22(record['trials'][call],record['input_preparation'][call],
                  OUTPUT/f'call_{call:02d}_diagnostics.json',context)
