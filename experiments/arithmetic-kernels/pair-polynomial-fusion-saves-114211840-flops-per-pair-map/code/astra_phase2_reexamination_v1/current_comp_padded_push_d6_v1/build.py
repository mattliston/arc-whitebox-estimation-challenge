"""Inert source-only padded-PUSH composition; output only under explicit --run."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
APUSH = NEW/'current_comp_aligned_push_d6_v1'
PARENT = APUSH/'build_v1/estimator.py'
PARENT_SHA = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
OUT = HERE/'build_v1'
OFFPY = P2/'data_gate/.venv/bin/python'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ARRAY_OPS = P2/'data_gate/.venv/lib/python3.11/site-packages/flopscope/_array_ops.py'
KERNEL = NEW/'current_d50_serial_d6_kernel_v1/kernel.py'
ARENA_ELEMENTS, PAD_START, PAD_END = 292225024, 118006528, 127443712
PAD_SPECS = ((118006528,119055104,(1024,1024)),(119055104,127443712,(1024,8192)))
OLD_INIT, ADDED_INIT, INIT = 1944970298, 9437184, 1954407482
SCHEMA = 'current_comp_padded_push_d6_build_v1'
SOURCE_PINS = {
    PARENT:PARENT_SHA,
    APUSH/'build.py':'ef008756a35e3e15e3f0286b6c846ea8d47c63a838d575bb01e54661cbe60f2b',
    APUSH/'DECISION.md':'ff026a7659875606a06206a89f047331a77d86af7da030266be74130bbf784f0',
    APUSH/'watch.py':'d6eaa5714b42836b530ae3db153dbef12fb2764f01d8403f917848b0d867bd91',
    APUSH/'PAD32_PRICE_20260910.md':'90f539405075c5513ca5e4dfde6c2782b427ee73bcd8a050711dd5facd721d9a',
    KERNEL:'a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7',
    ARRAY_OPS:'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    NEW/'check_remote_series_v1.py':'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55',
}
CHAINS = {
    APUSH/'build_v1':('f841d13473244938a24de14822529960455a2f344306d2899c6656fed4e44f0a',
        'e3d2ac39f6a2b0b489eb9a83beab987d2b36623e561d0611fe523fcdaacb0707',
        '9bac7e303cad3ec1a83045e16f18fc6874c40b6c2444372e1d890eb898502ffe'),
    APUSH/'runtime_v1':('5c1cb763e99882197d6c0fb6fdf5dc2e7e23b6ae100ac1d16ff60e5cf18b5d9e',
        '547cd2a9dc0a84ec7fff630886174ac736efa432e92c1f6009b351757c4cd724',
        '5cfbb63193d8f7e0da7cca38506a0449d92bfad4c4553b5443757dfe65679447'),
}
FIXTURE_PINS = {
    STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    STORE/'rows/observer_fit/0037/row_manifest.json':'900e3cc6929bcc81ea301479708fe8bdfb642447f6c8bb1a10981bffacc15571',
}
WEIGHT37_SHA = '967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4'
BIND_BEFORE = '        _p1_d6_832_bind(_SCRATCH[0])\n'
BIND_AFTER = BIND_BEFORE+'        _pad64_bind(_SCRATCH[0])\n'
DISPATCH_BRANCH = """    if (len(a.shape) == len(b.shape) == len(out.shape) == 2
            and int(a.shape[0]) == 1024 and 0 < int(a.shape[1]) <= 1024
            and int(a.shape[1]) % 64 == 32 and int(b.shape[1]) % 64 == 0
            and 1024 < int(b.shape[1]) <= 8192
            and int(b.shape[0]) == int(a.shape[1])
            and tuple(out.shape) == (1024,int(b.shape[1]))
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _SCRATCH is not None and arena_views is _SCRATCH[1]):
        return _pad64_product(a,b,out,arena_views)
"""
APPEND = '''

# BEGIN FIXED PAD32 PUSH-ONLY SERIAL-D6
_PAD64_SPECS = ((118006528,119055104,(1024,1024)),(119055104,127443712,(1024,8192)))
_PAD64_STATE = None
_PAD64_BUSY = False

def _pad64_bind(arena):
    global _PAD64_STATE
    assert _PAD64_STATE is None and not _PAD64_BUSY
    assert _SCRATCH is not None and arena is _SCRATCH[0]
    assert tuple(arena.shape) == (292225024,) and str(arena.dtype) == 'float32'
    assert _P1_D6_832_BINDINGS is not None
    binding = _P1_D6_832_BINDINGS[0]
    assert binding.arena is arena and binding.capacity == (1024,1024,8192) and not binding.busy
    assert _P1_D6_832_PUSH_INFO['required_elements'] == 118006528
    views = _cp3_bind_arena_views(arena,_PAD64_SPECS)
    zero = fnp.zeros((),dtype=fnp.float32)
    _PAD64_STATE = (arena,views,zero)

def _pad64_product(a,b,out,arena_views):
    global _PAD64_BUSY
    assert _SCRATCH is not None and arena_views is _SCRATCH[1]
    assert _PAD64_STATE is not None and not _PAD64_BUSY
    arena,views,zero = _PAD64_STATE
    binding = _P1_D6_832_BINDINGS[0]
    assert arena is _SCRATCH[0] and binding.arena is arena and not binding.busy
    assert binding.capacity == (1024,1024,8192)
    assert len(a.shape) == len(b.shape) == len(out.shape) == 2
    m,k,n = int(a.shape[0]),int(a.shape[1]),int(b.shape[1])
    assert m == 1024 and 0 < k <= 1024 and k % 64 == 32
    assert 1024 < n <= 8192 and n % 64 == 0
    assert tuple(b.shape) == (k,n) and tuple(out.shape) == (m,n)
    assert str(a.dtype) == str(b.dtype) == str(out.dtype) == str(zero.dtype) == 'float32'
    kp = k+32
    _PAD64_BUSY = True
    try:
        aa,bb = views[0][:m,:kp],views[1][:kp,:n]
        fnp.copyto(aa[:,:k],a)
        fnp.copyto(aa[:,k:kp],zero)
        fnp.copyto(bb[:k,:],b)
        fnp.copyto(bb[k:kp,:],zero)
        answer = _p1_d6_832_product(aa,bb,out,binding)
        assert answer is out and not binding.busy
        return out
    finally:
        _PAD64_BUSY = False
# END FIXED PAD32 PUSH-ONLY SERIAL-D6
'''


def require(ok, message):
    if not ok: raise RuntimeError(message)


def no_numerics():
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'No numerical imports')


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and path.suffix not in ('.npy','.npz','.f32')
        and 'truthpost' not in path.parts, 'Source/metadata only; no numerical-array reads')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick is not None: tick()
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.suffix == '.json' and not path.is_symlink() and path.stat().st_size < 4_000_000, 'Bounded metadata')
    return json.loads(path.read_text())


class Spans:
    """One immutable UTF8 byte/line index; AST columns are byte offsets."""
    def __init__(self, raw):
        self.raw = raw
        self.starts = [0]
        # LF defines Python source lines; UTF8 continuation bytes and formfeeds
        # must not create spurious line boundaries. CRLF remains byte-correct.
        for line in raw.split(b'\n')[:-1]: self.starts.append(self.starts[-1]+len(line)+1)

    def get(self, node):
        require(node.end_lineno is not None and node.end_col_offset is not None, 'Complete AST source span')
        start = self.starts[node.lineno-1]+node.col_offset
        end = self.starts[node.end_lineno-1]+node.end_col_offset
        require(0 <= start <= end <= len(self.raw), 'UTF8 AST span within immutable source')
        return self.raw[start:end].decode('utf-8')


def last_function(tree, name):
    found = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name]
    require(found, 'Existing active function '+name)
    return found[-1]


def assigned(tree, name):
    found = [n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == name for t in n.targets)]
    require(found, 'Existing metadata literal '+name)
    return found[-1]


def derive(tick=lambda:None):
    no_numerics()
    raw = PARENT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_SHA, 'Exact frozen APUSH parent bytes')
    text = raw.decode('utf-8'); require(text.encode('utf-8') == raw, 'Lossless source decoding')
    tick(); before = ast.parse(text); tick(); spans = Spans(raw)
    dispatcher = last_function(before,'_packed_product_4')
    old_dispatch = spans.get(dispatcher)
    tail = '    return _packed_product_cached(a,b,out,arena_views,4)'
    require(old_dispatch.count(tail) == 1 and old_dispatch.endswith(tail), 'Original final D4 fallback')
    new_dispatch = old_dispatch.replace(tail,DISPATCH_BRANCH+tail)
    require(text.count(old_dispatch) == text.count(BIND_BEFORE) == 1
        and '_PAD64_' not in text and '_pad64_' not in text, 'Only two unique original spans and new private names')
    changed = text.replace(old_dispatch,new_dispatch).replace(BIND_BEFORE,BIND_AFTER)+APPEND
    require(changed.endswith(APPEND), 'Exactly one inert appended block')
    inverse = changed[:-len(APPEND)]
    require(inverse.count(new_dispatch) == inverse.count(BIND_AFTER) == 1, 'Unique inverse spans')
    inverse = inverse.replace(BIND_AFTER,BIND_BEFORE).replace(new_dispatch,old_dispatch)
    require(inverse.encode('utf-8') == raw, 'Whole-source byte-exact inverse to APUSH')
    tick(); after = ast.parse(changed); tick(); added = ast.parse(APPEND)
    require(len(added.body) == 5 and all(isinstance(n,(ast.Assign,ast.FunctionDef)) for n in added.body)
        and all(not any(isinstance(x,ast.Call) for x in ast.walk(n.value))
            for n in added.body if isinstance(n,ast.Assign)), 'Only inert metadata/function declarations appended')
    require(ast.literal_eval(assigned(added,'_PAD64_SPECS')) == PAD_SPECS, 'Exact two padding view specs')
    restored = ast.Module(body=after.body[:-len(added.body)],type_ignores=after.type_ignores)
    for name in ('_packed_product_4','_predict_binary'):
        newer = last_function(restored,name)
        restored.body[restored.body.index(newer)] = last_function(before,name)
    before_dump = ast.dump(before,include_attributes=False); tick()
    restored_dump = ast.dump(restored,include_attributes=False); tick()
    require(before_dump == restored_dump, 'Only the two declared functions and inert helper block differ in the entire AST')
    predicted = spans.get(last_function(before,'_predict_binary'))
    require(predicted.count(BIND_BEFORE.rstrip('\n')) == 1 and text.count('_p1_d6_832_bind(_SCRATCH[0])') == 1,
        'Sole serial-layout cold binding in the active predictor')
    info = ast.literal_eval(assigned(before,'_P1_D6_832_PUSH_INFO'))
    require(info['capacity'] == (1024,1024,8192) and info['required_elements'] == PAD_START
        and max(stop for start,stop,shape in info['specs']) == PAD_START
        and all(0 <= start < stop <= PAD_START for start,stop,shape in info['specs'])
        and ast.literal_eval(assigned(before,'_PACKED_ELEMENTS')) == ARENA_ELEMENTS,
        'Exact live serial top/child owner ends before both new pads')
    require(PAD_SPECS[0][0] == PAD_START and PAD_SPECS[0][1] == PAD_SPECS[1][0]
        and PAD_SPECS[1][1] == PAD_END <= ARENA_ELEMENTS
        and sum(stop-start for start,stop,shape in PAD_SPECS) == ADDED_INIT
        and all(stop-start == shape[0]*shape[1] for start,stop,shape in PAD_SPECS)
        and OLD_INIT+ADDED_INIT == INIT, 'Disjoint same-arena spans and exact public cold reshape tariff')
    calls = [n for n in ast.walk(before) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id == '_packed_product_4']
    mm = last_function(before,'_mm'); mm_calls = [n for n in ast.walk(mm) if n in calls]
    require(len(calls) == 3 and len(mm_calls) == 1 and all(len(n.args) == 4 for n in calls),
        'Only active _mm and the two square-source callsites reach final dispatcher')
    for call in calls:
        if call not in mm_calls:
            require(ast.dump(call.args[3],include_attributes=False) ==
                ast.dump(ast.parse('_SCRATCH[4]',mode='eval').body,include_attributes=False),
                'All non-_mm callers use a different square owner and fail the padded predicate')
    mm_source = spans.get(mm)
    require("out = fnp.empty((mp, npad), dtype=fnp.float32)\n    out = _packed_product_4(aa, bb, out," in mm_source
        and "_SCRATCH[1] if n > 1024 else _SCRATCH[2]" in mm_source,
        'The only admissible caller supplies its own fresh output and original PUSH owner')
    binder = spans.get(last_function(before,'_cp3_bind_arena_views'))
    require('fnp.reshape' in binder, 'Existing public counted view binder')
    compile(changed,str(OUT/'estimator.py'),'exec'); tick(); no_numerics()
    proof = dict(parent_sha256=PARENT_SHA,candidate_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole_source_inverse_verified=True,entire_AST_change_scope_verified=True,
        unchanged_imports_literals_heads_physics_and_kernel_arithmetic=True,
        original_aligned_branch_byte_unchanged=True,no_read_or_square_change=True,
        same_original_backing_arena=True,backing_arena_bytes=4*ARENA_ELEMENTS,
        serial_required_elements=PAD_START,padding_specs=PAD_SPECS,padding_region_bytes=4*ADDED_INIT,
        padding_region_disjoint_from_all_live_serial_views=True,original_caller_output_retained=True,
        square_callers_excluded_by_owner_identity=True,original_serial_busy_and_child_bypass_unchanged=True,
        padding_busy_guard_and_full_per_call_overwrite=True,no_new_full_backing_array=True,
        no_import_time_array_work=True,extra_cold_reshape_calls=2,extra_cold_FP32_zero_calls=1,
        extra_cold_binding_flops=ADDED_INIT,expected_complete_initialization_flops=INIT,
        per_padded_call_added_copy_flops='(k+32)*(1024+n)',per_padded_call_copy_calls=4,
        new_policy='FP32 original PUSH owner; m1024, k%64=32, 0<k<=1024, 1024<n<=8192, n%64=0; kp=k+32',
        parameter_updates=0,estimator_executed=False,raw_error_measured=False)
    return changed,proof


def admission(args,tick):
    sources = {str(p):v for p,v in SOURCE_PINS.items()}
    sources.update({str(Path(__file__).resolve()):args.source_sha256,str(HERE/'PROTOCOL.md'):args.protocol_sha256})
    inputs = {str(p):v for p,v in FIXTURE_PINS.items()}
    for folder,digests in CHAINS.items():
        paths = [folder/name for name in ('result.json','completion.json','finalization.json')]
        for path,digest in zip(paths,digests): inputs[str(path)] = digest
        record,end,final = [read(p) for p in paths]
        require(all(x.get('passed') is True and x.get('completed') is True for x in (record,end,final))
            and end['result_sha256'] == final['result_sha256'] == digests[0]
            and final['completion_sha256'] == digests[1] and record['candidate_sha256'] == PARENT_SHA,
            'Actual direct parent build/runtime three seals')
        if folder.name == 'runtime_v1':
            require(record['schema'] == 'current_comp_aligned_push_d6_runtime_v1'
                and all(record[k] is True for k in ('runtime_falsifier_passed','repeat_byte_exact','sources_unchanged',
                    'hashes_unchanged','whole_owned_tree_retired','cleanup_passed','observational_groups_retired'))
                and record['initialization_flops'] == OLD_INIT and record['ranks'] == [35,35]
                and record['full_model_calls'] == 2 and record['maximum_swap_growth_bytes'] == 0
                and 0 < record['maximum_combined_rss_bytes'] < 8_000_000_000
                and 0 < record['maximum_interval_s'] <= 1 and final['elapsed_s'] < 300,
                'Limited successful APUSH runtime authority, not a new score or37 runtime')
        else:
            require(record['schema'] == 'current_comp_aligned_push_d6_build_v1'
                and record['proof']['whole_source_inverse_verified']
                and record['proof']['existing_complete_initialization_flops'] == OLD_INIT,
                'Exact already-built APUSH model/kernel owner')
    for path,digest in {**sources,**inputs}.items(): require(sha(path,tick) == digest,'Immutable direct source/metadata binding')
    rows = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    require(rows[37]['role_rank'] == 37 and rows[37]['path'] == 'rows/observer_fit/0037/row_manifest.json'
        and rows[37]['sha256'] == FIXTURE_PINS[STORE/'rows/observer_fit/0037/row_manifest.json']
        and read(STORE/'rows/observer_fit/0037/row_manifest.json')['files']['weights.npy']['sha256'] == WEIGHT37_SHA,
        'Production client list-index37 equals the exact current canonical native37 fixture')
    return sources,inputs


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def build(args):
    require(args.run,'No output without explicit --run')
    began = time.monotonic(); no_numerics()
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.dont_write_bytecode
        and sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0) >= 10,'Official source-only nice10/bytecode-off/assertions')
    require(not OUT.exists() and not HERE.is_symlink(),'Exclusive build; no overwrite or retry')
    record = dict(schema=SCHEMA,passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
        parameter_updates=0,estimator_executed=False,estimator_imported=False,truth_arrays_accessed=False,
        truth_accessed=False,score_measured=False,numerical_imports=False,source_bindings={},input_bindings={},artifacts={})
    def guard(*unused):
        require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
            'Entire source-only build60s/2GB')
    handler = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    OUT.mkdir()
    try:
        sources,inputs = admission(args,guard)
        record.update(source_bindings=sources,input_bindings=inputs)
        candidate,proof = derive(guard); guard()
        path = OUT/'estimator.py'
        with path.open('x',encoding='utf-8',newline='') as stream: stream.write(candidate)
        require(sha(path,guard) == proof['candidate_sha256'],'Exact generated source readback')
        record.update(candidate_sha256=proof['candidate_sha256'],proof=proof,artifacts={str(path):proof['candidate_sha256']})
        for p,digest in {**sources,**inputs,**record['artifacts']}.items(): require(sha(p,guard) == digest,'Fresh complete source/input/output rehash')
        no_numerics(); guard()
        record.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        record.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,handler)
        record.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if record['elapsed_s'] >= 60 or record['peak_rss_bytes'] >= 2_000_000_000:
            record.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False)
        write(OUT/'result.json',record); rh = sha(OUT/'result.json')
        ok = bool(record['passed'] and time.monotonic()-began < 60)
        write(OUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(OUT/'completion.json')
        elapsed = time.monotonic()-began; ok = bool(ok and elapsed < 60)
        write(OUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
        sha(OUT/'finalization.json'); ok = bool(ok and time.monotonic()-began < 60)
    print(json.dumps(dict(passed=ok,candidate_sha256=record.get('candidate_sha256'),error=record.get('error'))),flush=True)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256'); parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return 0
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in (args.source_sha256,args.protocol_sha256)),
        'Explicit reviewed builder/protocol authority')
    return build(args)


if __name__ == '__main__': raise SystemExit(main())
