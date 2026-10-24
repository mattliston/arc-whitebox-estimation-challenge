"""Fixed, target-blind native37/repeat37; explicit launch and source bindings only."""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import textwrap
import time
import traceback
import types
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
APUSH = NEW/'current_comp_aligned_push_d6_v1'
LEGACY = NEW/'birth_direction_fit_v1/native_v2.py'
LEGACY_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
OBSERVER = APUSH/'native.py'
OBSERVER_SHA = '751bc155506b2825a8c0d27553e8ced8d7db66d68434247bb158c22fad3c3ad6'
PARENT_SHA = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
CONTROL_SHA = '23b88f12449442cad00eb352e2aca1d90d277120b7ed348f800a9bcff64684a7'
CONTROL_WATCH_SHA = '8665b4705c07d14f7f315ee1e1229154177fc86e3345f12d7029fe31ed8b526b'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ROW = STORE/'rows/observer_fit/0037'
FIXTURE = {
    STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    ROW/'row_manifest.json':'900e3cc6929bcc81ea301479708fe8bdfb642447f6c8bb1a10981bffacc15571',
    ROW/'weights.npy':'967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4',
}
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
BUDGET, CAP, RSS = 2**41, 300., 8_000_000_000
KEYS = ('source','watch','protocol','watch_spec','compose','fnp','design')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npz','.npy') and 'truth' in path.name.lower()),
            'Target-blind acquisition never opens truth arrays')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            if tick is not None:
                tick()
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    def safe(x):
        if isinstance(x,float) and not math.isfinite(x):
            return dict(nonfinite_float=repr(x))
        if isinstance(x,dict):
            return {k:safe(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):
            return [safe(v) for v in x]
        return x
    with Path(path).open('x') as stream:
        json.dump(safe(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load(path, name, digest):
    require(sha(path) == digest, 'Frozen source: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def environment():
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
        and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS, 0) >= 10
        and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Official nice10/BLAS1/assertions/no fallback')
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Stdlib-only admission')
    versions = {k: importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
    require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),
            'Original official package versions')


def source_owners(args):
    mapping = {HERE/'probe.py':args.source_sha256,HERE/'watch_probe.py':args.watch_sha256,
        HERE/'PROBE_PROTOCOL.md':args.protocol_sha256,HERE/'PROBE_WATCH_SPEC.md':args.watch_spec_sha256,
        HERE/'compose.py':args.compose_sha256,HERE/'fnp.py':args.fnp_sha256,
        HERE/'PROTOCOL.md':args.design_sha256,LEGACY:LEGACY_SHA,OBSERVER:OBSERVER_SHA}
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in mapping.values()), 'Explicit reviewed owners')
    return {str(p):v for p,v in mapping.items()}


def bind(mapping, tick=None):
    for path, digest in mapping.items():
        require(sha(path, tick) == digest, 'Immutable input/source: '+str(path))


def controls(inputs, tick=None):
    """Direct completed native40 authority, never rerun or relabel older receipts."""
    folder, watched = APUSH/'native40_terminal_v1', APUSH/'native40_watch_v1'
    chains = []
    for directory, expected in ((folder,CONTROL_SHA),(watched,CONTROL_WATCH_SHA)):
        paths = [directory/name for name in ('result.json','completion.json','finalization.json')]
        digests = [sha(p,tick) for p in paths]
        require(digests[0] == expected, 'Pinned positive control boundary')
        records = [read(p) for p in paths]
        require(all(x.get('passed') is True and x.get('completed') is True for x in records)
            and records[1]['result_sha256'] == records[2]['result_sha256'] == digests[0]
            and records[2]['completion_sha256'] == digests[1], 'Actual control three seals')
        inputs.update({str(p):d for p,d in zip(paths,digests)})
        chains.append((records,digests))
    (record,_,_), ds = chains[0]
    (watch,_,_), ws = chains[1]
    require(record['candidate_sha256'] == watch['candidate_sha256'] == PARENT_SHA
        and record['full_model_calls'] == record['retired_call_count'] == watch['full_model_calls'] == 41
        and [watch['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')] == ds
        and watch['whole_owned_tree_retired'] and watch['cleanup_passed']
        and watch['maximum_combined_rss_bytes'] < RSS and watch['maximum_interval_s'] <= 1
        and watch['maximum_swap_growth_bytes'] == 0, 'Completed own-cost/domain/control acquisition')
    selected = []
    for index, expected in ((0,461231998786),(40,459287028488)):
        trial = record['trials'][index]
        require(trial['rank'] == 37 and trial['call'] == index and trial['flops'] == expected
            and trial['includes_initialization'] is (index == 0)
            and trial['multiplier'] == max(.1,expected/BUDGET), 'Same physical cold/repeat37 control')
        for path in (trial['output'],str(folder/f'call_{index:02d}_bill.json'),trial['input_bill'],
                trial['input_reset_receipt'],trial['lifecycle_receipt'],str(folder/f'call_{index:02d}_diagnostics.json')):
            digest = record['artifacts'][path]
            require(sha(path,tick) == digest, 'Selected control artifact')
            inputs[path] = digest
        require(read(trial['input_bill'])['weight_file_sha256'] == FIXTURE[ROW/'weights.npy'],
                'Control uses the identical canonical weights')
        selected.append(trial)
    return selected, dict(native=ds,watch=ws)


def paid_loop_source():
    """Original complete paid input/predict/save/reset/release loop, not rewritten billing."""
    original = LEGACY.read_text()
    require(sha(LEGACY) == LEGACY_SHA, 'Original input/prediction lifecycle owner')
    main = next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    nodes = [n for n in ast.walk(main) if isinstance(n,ast.With)
             and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name)
                     and i.context_expr.func.id == 'threadpool_limits' for i in n.items)]
    require(len(nodes) == 1, 'One exact original BLAS/paid loop')
    node = nodes[0]
    source = textwrap.dedent(''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno]))
    before = source
    edits = [
        ('            served=estimator.predict(mlp,BUDGET)',
         '            observer.begin(); observer.ctx=ctx\n'
         '            try:\n'
         '                served=observed_predict(estimator,mlp,observer)\n'
         '            finally:\n'
         '                observer.ctx=None'),
        ('        if call==0: first_bytes=values.tobytes()',
         '        finish_call(np,observer,call,rank,result,args.out,trial)\n'
         '        if call==0: first_bytes=values.tobytes()'),
        ('        del mlp,served,materialized,values\n        gc.collect()',
         '        del mlp,served,materialized,values\n        gc.collect()\n'
         '        retire_call(call,rank,result,args.out,trial)')]
    for old,new in edits:
        require(source.count(old) == 1, 'Unique passive observation splice')
        source = source.replace(old,new)
    inverse = source
    for old,new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique inverse observation splice')
        inverse = inverse.replace(new,old)
    require(inverse == before, 'Whole paid-loop inverse byte-exact')
    ast.parse(source)
    return source, dict(owner_sha256=LEGACY_SHA,original_loop_sha256=hashlib.sha256(before.encode()).hexdigest(),
        derived_loop_sha256=hashlib.sha256(source.encode()).hexdigest(),whole_loop_inverse=True,
        no_billing_input_reset_materialization_or_repeat_change=True)


def observers(owner):
    (domain, comp, extra), proof = owner.observer_sources()
    env = dict(c=types.SimpleNamespace(require=require), require=require)
    exec(compile(domain, '<unchanged-domain-observer>', 'exec'),env)
    env.update(Domain=env['DomainObserver'],bound_depths=owner.base().bound_depths,
        EXTRA_LAYOUTS=owner.EXTRA_LAYOUTS,EXTRA_BINDINGS=owner.EXTRA_BINDINGS,
        eligible=owner.eligible,product_flops=owner.product_flops)
    exec(compile(comp+'\n'+extra, '<unchanged-APUSH-passive-observer>', 'exec'),env)
    ds = owner.TL.joinpath('native.py').read_text()
    exec(compile(owner.function_source(ds,'decode')+'\n'+owner.function_source(ds,'diagnostics'),
                 '<unchanged-all16-all47-checks>', 'exec'),env)
    env['hashlib'] = hashlib
    return env['Observer'], env['diagnostics'], proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    environment(); began = time.monotonic()
    args.out = HERE/'probe_v1'
    require(not args.out.exists() and not HERE.is_symlink(), 'Exclusive never-resumed output')
    args.out.mkdir()
    result = dict(schema='current_comp_shared_old384_native37_v1',passed=False,completed=False,
        source_bindings=source_owners(args),input_bindings={},artifacts={},trials=[],input_preparation=[],
        diagnostic_records=[],truth_accessed=False,score_measured=False,full_model_calls=0,
        retired_call_count=0,production_residual_qualified=False,parameter_updates=0)
    def tick():
        require(time.monotonic()-began < CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < RSS,
                'Whole native stage300s/process8GB; external watch owns live full-tree cap')
    try:
        bind(result['source_bindings'],tick)
        composition = load(HERE/'compose.py','shared384_composition',args.compose_sha256)
        result['source_bindings'].update({str(p):v for p,v in composition.authenticate().items()})
        text, proof = composition.derive_source()
        result['composition_proof'] = proof
        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        owner = load(OBSERVER,'shared384_observer_owner',OBSERVER_SHA)
        result['source_bindings'].update({str(p):v for p,v in owner.OWNERS.items()})
        legacy = load(LEGACY,'shared384_lifecycle_owner',LEGACY_SHA)
        result['source_bindings'].update({str(p):v for p,v in legacy.PINS.items()})
        Observer,diagnostics,op = observers(owner)
        result['passive_observer_source_proof'] = op
        loop,lp = paid_loop_source(); result['paid_loop_proof'] = lp
        result['input_bindings'].update({str(p):v for p,v in FIXTURE.items()})
        result['controls'],result['control_seals'] = controls(result['input_bindings'],tick)
        bind(result['source_bindings'],tick); bind(result['input_bindings'],tick); environment()
        row = read(ROW/'row_manifest.json')
        rows = {37:dict(weights=str(ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}
        result['ranks'] = [37,37]
        result['started_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.out/'started.json',result)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = types.ModuleType('shared384_prototype')
        module.__file__ = '<shared-old384-composed-'+result['candidate_sha256']+'>'
        exec(compile(text,module.__file__,'exec'),vars(module))
        observer = Observer(module)
        def finish_call(np,observer,call,rank,result,folder,trial):
            packet,startup,dispatches = observer.finish()
            domain = diagnostics(np,packet)
            record = dict(call=call,rank=rank,domains=domain,startup=startup,dispatches=dispatches)
            path = folder/f'call_{call:02d}_diagnostics.json'; write(path,record)
            digest = sha(path); result['artifacts'][str(path)] = digest
            if result['diagnostic_records']:
                before = read(result['diagnostic_records'][0]['path'])
                require(domain == before['domains'], 'Exact repeat domain and ordered selectors')
            result['diagnostic_records'].append(dict(path=str(path),sha256=digest))
            result['full_model_calls'] = call+1
            require(startup['original_arena_bytes'] == 1168900096 and startup['no_second_backing_arena'],
                    'Original packed-kernel arena and lifetime preserved')
            del packet
            tick()
        def retire_call(call,rank,result,folder,trial):
            life = read(trial['lifecycle_receipt'])
            require(life['context_released'] and life['closed_context_bill_unchanged_by_reset']
                and life['complete'] and observer.ctx is None and call == result['retired_call_count'],
                'Actual saved context retirement and ordered fixed call')
            record = dict(passed=True,completed=True,call=call,rank=rank,context_retired=True,
                bill_sha256=sha(folder/f'call_{call:02d}_bill.json'),
                lifecycle_sha256=sha(trial['lifecycle_receipt']),
                diagnostic_sha256=result['diagnostic_records'][call]['sha256'],
                output_sha256=trial['output_sha256'],elapsed_s=time.monotonic()-began)
            path = folder/f'call_{call:02d}_complete.json'; write(path,record)
            result['artifacts'][str(path)] = sha(path)
            result['retired_call_count'] += 1
            print(json.dumps(dict(event='retired',call=call,rank=rank,complete_sha256=sha(path))),flush=True)
            tick()
        env = dict(vars(legacy),np=np,flops=flops,fnp=fnp,threadpool_limits=threadpool_limits,
            threadpool_info=threadpool_info,MLP=MLP,derive_estimator_seed=derive_estimator_seed,
            materialise_predictions=materialise_predictions,args=args,inputs=result['input_bindings'],
            rows=rows,ranks=(37,),result=result,module=module,estimator=None,first_bytes=None,
            observer=observer,observed_predict=owner.base().observed_predict,
            finish_call=finish_call,retire_call=retire_call)
        exec(compile(loop,'<exact-native-paid-loop-with-passive-observation>','exec'),env)
        require(len(result['trials']) == len(result['diagnostic_records']) == 2, 'Both complete fixed calls')
        # The first affected pre-state is layer5. Earlier predictions and all
        # learned heads must therefore reproduce the authentic parent exactly.
        prefix_checks = []
        for trial,control in zip(result['trials'],result['controls']):
            old = np.load(control['output'],allow_pickle=False)
            new = np.load(trial['output'],allow_pickle=False)
            require(old.dtype == new.dtype == np.float32 and old.shape == new.shape == (16,1024)
                and np.isfinite(old).all() and np.isfinite(new).all()
                and old[:5].tobytes() == new[:5].tobytes(), 'Unaffected five-layer current-bank/head byte parity')
            prefix_checks.append(dict(call=trial['call'],layers=5,byte_exact=True))
            del old,new
        result['unaffected_prefix_checks'] = prefix_checks
        result['initialization_flops'] = result['trials'][0]['flops']-result['trials'][1]['flops']
        require(result['initialization_flops'] >= 1944970298, 'Original startup is fully owned, no subtraction')
        result['full_model_calls'] = result['retired_call_count'] = 2
        del env,observer,module
        gc.collect()
        bind(result['source_bindings'],tick); bind(result['input_bindings'],tick); bind(result['artifacts'],tick)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            all16_domains_qualified=True,all47_selectors_qualified=True,repeat_byte_exact=True,
            own_complete_bills=True,input_preparation_in_participant_bill=False)
    except BaseException:
        result['error'] = traceback.format_exc()
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if result['elapsed_s'] >= CAP or result['peak_rss_bytes'] >= RSS:
        result.update(passed=False,completed=False)
    write(args.out/'result.json',result); rh = sha(args.out/'result.json')
    ok = bool(result['passed'] and time.monotonic()-began < CAP)
    write(args.out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
    ch = sha(args.out/'completion.json'); elapsed = time.monotonic()-began
    ok = bool(ok and elapsed < CAP)
    write(args.out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    sha(args.out/'finalization.json'); ok = bool(ok and time.monotonic()-began < CAP)
    print(json.dumps(dict(passed=ok,completed=ok,full_model_calls=result['full_model_calls'],
        elapsed_s=result['elapsed_s'],initialization_flops=result.get('initialization_flops'),error=result.get('error'))),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
