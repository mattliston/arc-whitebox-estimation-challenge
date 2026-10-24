"""Inert, target-blind v29 cold37/repeat37 acquisition; no learned head."""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
SITE = P2/'data_gate/.venv/lib/python3.11/site-packages'
OFFPY = P2/'data_gate/.venv/bin/python'
LOOP_OWNER = NEW/'current_comp_shared_old384_v1/probe.py'
LOOP_SHA = '6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b'
WATCH_OWNER = NEW/'current_comp_shared_old384_v1/watch_probe.py'
WATCH_OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'
ADAPTER_SHA = 'cab844cd7981d31c7e44f4797ad60490729bdfadb8ce46655cdd5966ae9f36b7'
ADAPTER_PROTOCOL_SHA = '2a93907e69427a55c9a11eef7bd2143d9f71835d62f5eb4323700c1a8bd545ba'
SDK = SITE/'whestbench/sdk.py'
SDK_SHA = 'b0fcc52c6b531981e46da6955365aa786260fab53fd66dcf3675791ed8c3c105'
POINTWISE = SITE/'flopscope/_pointwise.py'
POINTWISE_SHA = '7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7'
SOURCE_PINS = {
    LOOP_OWNER: LOOP_SHA, WATCH_OWNER: WATCH_OWNER_SHA,
    HERE/'adapter.py': ADAPTER_SHA, HERE/'PROTOCOL.md': ADAPTER_PROTOCOL_SHA,
    SDK: SDK_SHA, POINTWISE: POINTWISE_SHA,
    SITE/'whestbench/runner.py': 'e2a3db512249fa705775d27b5956971110872958e0603d23bcf24f2673e908fa',
    SITE/'whestbench/subprocess_worker.py': '577744e2241a09dbc45ba244c4adca9c72bdbbccdc0ec92fea81197df4300971',
    SITE/'whestbench/cli.py': 'f6b975fd39a1dfda5e71efada4f7f4da0b5fde0b0130816484a660f91dd2b58d',
}
CAP, RSS, BUDGET = 300., 8_000_000_000, 2**41
KEYS = ('source','watch','protocol','watch_spec','adapter','adapter_protocol')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No target-array access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed source owner '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def environment():
    return load(LOOP_OWNER, 'v29_native_environment_owner', LOOP_SHA).environment()


def bind(mapping, tick=None):
    for path, digest in mapping.items():
        require(sha(path, tick) == digest, 'Frozen source/input/artifact '+str(path))


def write(path, value):
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x):
            return {'nonfinite_float': repr(x)}
        if isinstance(x, dict):
            return {k:safe(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)):
            return [safe(v) for v in x]
        return x
    with Path(path).open('x') as stream:
        json.dump(safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source_owners(args):
    require(args.adapter_sha256 == ADAPTER_SHA and args.adapter_protocol_sha256 == ADAPTER_PROTOCOL_SHA,
            'The original feature adapter and feature specification remain frozen')
    values = {HERE/'worker.py':args.source_sha256, HERE/'watch.py':args.watch_sha256,
        HERE/'ACQUISITION_PROTOCOL.md':args.protocol_sha256, HERE/'WATCH_SPEC.md':args.watch_spec_sha256,
        **SOURCE_PINS}
    require(all(isinstance(h,str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                for h in values.values()), 'Explicit reviewed source hashes')
    return {str(p):h for p,h in values.items()}


def paid_loop_source(owner):
    original, prior = owner.paid_loop_source()
    before = '                    if estimator is None: estimator=module.Estimator()'
    # The extracted loop is dedented eight columns by its original owner.
    before = before[8:]
    after = '            if estimator is None: estimator=cold_create(module,ctx,result)'
    require(original.count(before) == 1, 'Unique paid cold construction site')
    changed = original.replace(before,after)
    require(changed.count(after) == 1 and changed.replace(after,before) == original,
            'Whole paid-loop inverse, sole explicit setup addition')
    return changed, dict(parent=prior, setup_splice_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_owned_inputs_bills_materialization_resets_release_and_repeat=True)


def variance_sites(source):
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'Estimator']
    require(len(classes) == 1, 'Exact v29 estimator class')
    core = [n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name == '_predict_core']
    require(len(core) == 1, 'Exact v29 physical core')
    calls = []
    for node in ast.walk(core[0]):
        if (isinstance(node,ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id == 'var'
                and isinstance(node.value,ast.Call)):
            call = node.value
            if (isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name)
                    and call.func.value.id == 'fnp' and call.func.attr == 'maximum'):
                require(len(call.args) == 2 and isinstance(call.args[1],ast.Constant)
                        and call.args[1].value == 1e-10 and not call.keywords, 'Original variance floor')
                calls.append((call.lineno,call.end_lineno))
    require(len(calls) == 4, 'All four physical pre-floor sites, not feature-hook var')
    return tuple(sorted(calls))


class Capture:
    """Only detached bytes; never retain a frame or a pooled array."""
    def __init__(self, module, adapter, sites):
        self.module, self.adapter, self.sites = module, adapter, sites
        self.core = module.Estimator._predict_core.__code__
        self.maximum = module.fnp.maximum
        self.maximum_code = self.maximum.__code__
        self.begin()

    def begin(self):
        self.ctx = None
        self.features, self.variances = [], []

    @staticmethod
    def detach(value):
        return dict(data=value.tobytes(), dtype=str(value.dtype), shape=tuple(value.shape))

    def __call__(self, frame, event, unused):
        if event != 'call' or frame.f_code is not self.maximum_code:
            return
        caller = frame.f_back
        if caller is None or caller.f_code is not self.core or frame.f_locals.get('wrapped') is not self.maximum:
            return
        if not any(lo <= caller.f_lineno <= hi for lo,hi in self.sites):
            return
        args, kwargs = frame.f_locals['args'], frame.f_locals['kwargs']
        require(len(args) == 2 and args[1] == 1e-10 and not kwargs, 'Actual unchanged floor operands')
        value, layer = args[0], int(caller.f_locals['li'])
        require(layer == len(self.variances) and tuple(value.shape) == (1024,)
                and str(value.dtype) == 'float32', 'Ordered actual FP32 pre-floor vector')
        before = (int(self.ctx.flops_used),len(self.ctx.op_log))
        self.variances.append(dict(layer=layer, **self.detach(value)))
        require((int(self.ctx.flops_used),len(self.ctx.op_log)) == before, 'Passive pre-floor bytes only')

    def sink(self, packet):
        require(packet['layer'] == len(self.features), 'One fixed feature packet per layer')
        saved = {k:v for k,v in packet.items() if k not in ('features','host','sigma')}
        saved.update({k:self.detach(packet[k]) for k in ('features','host','sigma')})
        self.features.append(saved)

    def predict(self, estimator, mlp):
        require(sys.getprofile() is None and self.ctx is not None, 'Exclusive passive profiler and actual context')
        self.adapter.attach_capture(estimator,self.sink,self.ctx)
        sys.setprofile(self)
        try:
            return estimator.predict(mlp,BUDGET)
        finally:
            sys.setprofile(None)
            self.adapter.detach_capture(estimator)

    def export(self):
        require(self.ctx is None and len(self.features) == len(self.variances) == 16,
                'Complete16 feature and actual pre-floor packets after context release')
        answer = self.features, self.variances
        self.features, self.variances = [], []
        return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    environment(); began = time.monotonic()
    args.out = HERE/'acquisition_v1'
    require(not args.out.exists() and not HERE.is_symlink(), 'Exclusive never-resumed acquisition')
    args.out.mkdir()
    result = dict(schema='current_v29_temporal_acquisition_v1',passed=False,completed=False,
        source_bindings=source_owners(args),input_bindings={},artifacts={},trials=[],input_preparation=[],
        capture_records=[],full_model_calls=0,retired_call_count=0,parameter_updates=0,
        truth_accessed=False,score_measured=False,head_attached=False,feedback_enabled=False,
        feedback_scope='learned observer only; original physical F68/F69 feedback unchanged',
        production_residual_qualified=False,setup_calls=0,setup_seed=0)
    def tick():
        require(time.monotonic()-began < CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < RSS,
                'Whole300s/worker8GB; watcher owns full-tree resources')
    try:
        bind(result['source_bindings'],tick)
        owner = load(LOOP_OWNER,'v29_native_loop_owner',LOOP_SHA)
        adapter = load(HERE/'adapter.py','v29_frozen_capture_adapter',ADAPTER_SHA)
        result['source_bindings'].update(adapter.authenticate())
        legacy = load(owner.LEGACY,'v29_exact_paid_input_owner',owner.LEGACY_SHA)
        result['source_bindings'].update({str(p):v for p,v in legacy.PINS.items()})
        result['source_bindings'][str(owner.LEGACY)] = owner.LEGACY_SHA
        result['input_bindings'].update({str(p):h for p,h in owner.FIXTURE.items()})
        source, proof = adapter.derive_source()
        sites = variance_sites(source)
        result.update(composition_proof=proof,candidate_sha256=proof['derived_sha256'],
                      raw_variance_sites=sites,ranks=[37,37])
        loop, result['paid_loop_proof'] = paid_loop_source(owner)
        row = owner.read(owner.ROW/'row_manifest.json')
        rows = {37:dict(weights=str(owner.ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))}
        bind(result['source_bindings'],tick); bind(result['input_bindings'],tick); environment()
        write(args.out/'started.json',result)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.sdk import SetupContext
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module, actual_proof = adapter.load_runtime(allow_numerical_imports=True)
        require(actual_proof == proof and module.NO_CORR and module.STRASSEN_FIRST == module.STRASSEN_LEVELS == 5,
                'Exact frozen head-free physical source/configuration')
        observer = Capture(module,adapter,sites)
        def cold_create(module,ctx,result):
            start, count, began_setup = int(ctx.flops_used),len(ctx.op_log),time.monotonic()
            estimator = module.Estimator()
            setup = SetupContext(width=1024,depth=16,flop_budget=BUDGET,api_version='1.0',
                                 submission_dir=str(adapter.OWNER.parent),seed=0)
            estimator.setup(setup)
            elapsed = time.monotonic()-began_setup
            result['setup'] = dict(seed=0,api_version='1.0',flops=int(ctx.flops_used)-start,
                operation_count=len(ctx.op_log)-count,wall_s=elapsed,inside_cold_participant_bill=True,
                constructor_included=True,original_warmup_exception_policy_unchanged=True)
            result['setup_calls'] += 1
            require(result['setup_calls'] == 1 and elapsed < 5, 'One unchanged setup, official5s cap')
            return estimator
        def finish_call(np,observer,call,rank,result,folder,trial):
            packets, raw = observer.export()
            arrays, fields, domains = {},{},[]
            served = np.load(trial['output'],allow_pickle=False)
            for layer,(packet,pre) in enumerate(zip(packets,raw)):
                require(packet['layer'] == pre['layer'] == layer, 'Exact captured layer alignment')
                for key, item in {**{k:packet[k] for k in ('features','host','sigma')},'prevariance':pre}.items():
                    value = np.frombuffer(item['data'],dtype=item['dtype']).reshape(item['shape'])
                    expected = (1024,30) if key == 'features' else (1024,)
                    require(value.dtype == np.float32 and value.shape == expected and np.isfinite(value).all(),
                            'Finite actual detached FP32 capture '+key)
                    name = f'layer_{layer:02d}_{key}'
                    arrays[name] = value
                    fields[name] = dict(dtype=str(value.dtype),shape=list(value.shape),
                                        raw_sha256=hashlib.sha256(item['data']).hexdigest())
                X, v = arrays[f'layer_{layer:02d}_features'], arrays[f'layer_{layer:02d}_prevariance']
                require(float(v.min()) >= 1e-10 and (v < 1e-10).sum() == 0, 'Unfloored actual physical domain')
                require(arrays[f'layer_{layer:02d}_host'].tobytes() == served[layer].tobytes(),
                        'Head-free captured host equals actual served row')
                zero_slots = adapter.TERMINAL_ZERO_SLOTS if layer == 15 else adapter.SKEW_ZERO_FIRST_SLOTS if layer == 0 else ()
                require(not zero_slots or np.all(X[:,zero_slots] == 0), 'Unavailable moments have exact zero features')
                domains.append(dict(layer=layer,minimum=float(v.min()),below_floor_count=0,
                    raw_sha256=fields[f'layer_{layer:02d}_prevariance']['raw_sha256'],finite=True))
            path = folder/f'call_{call:02d}_capture.npz'
            with path.open('xb') as stream:
                np.savez(stream,**arrays)
            result['artifacts'][str(path)] = sha(path)
            feature_f = sum(p['feature_flops'] for p in packets)
            feature_ops = sum(p['feature_operations'] for p in packets)
            record = dict(call=call,rank=rank,path=str(path),sha256=sha(path),arrays=fields,
                domains=domains,feature_flops=feature_f,feature_operations=feature_ops,
                layer_feature_flops=[p['feature_flops'] for p in packets],
                layer_feature_operations=[p['feature_operations'] for p in packets],
                feature_F_in_actual_bill=True,host_served_byte_exact=True,all16_raw_domains=True,
                head_attached=False,feedback_enabled=False,
                feedback_scope='learned observer only; original physical F68/F69 feedback unchanged')
            require(0 < feature_f <= 16*adapter.feature_ceiling() and feature_f < trial['flops'], 'Complete paid feature subtotal')
            if result['capture_records']:
                first = result['capture_records'][0]
                require(fields == first['arrays'] and domains == first['domains']
                    and record['layer_feature_flops'] == first['layer_feature_flops']
                    and record['layer_feature_operations'] == first['layer_feature_operations'], 'Exact repeat features/domains/tariffs')
            diag = folder/f'call_{call:02d}_diagnostics.json'; write(diag,record)
            result['artifacts'][str(diag)] = sha(diag)
            result['capture_records'].append(record); result['full_model_calls'] = call+1
            del packets,raw,arrays,served,value,X,v
            gc.collect(); tick()
        def retire_call(call,rank,result,folder,trial):
            life = owner.read(trial['lifecycle_receipt'])
            require(life['context_released'] and life['closed_context_bill_unchanged_by_reset'] and life['complete']
                and observer.ctx is None and not observer.features and not observer.variances
                and sys.getprofile() is None and call == result['retired_call_count'], 'Released paid context/capture/profile')
            record = dict(passed=True,completed=True,call=call,rank=rank,context_retired=True,
                bill_sha256=sha(folder/f'call_{call:02d}_bill.json'),lifecycle_sha256=sha(trial['lifecycle_receipt']),
                diagnostic_sha256=sha(folder/f'call_{call:02d}_diagnostics.json'),
                capture_sha256=result['capture_records'][call]['sha256'],output_sha256=trial['output_sha256'],
                elapsed_s=time.monotonic()-began)
            path = folder/f'call_{call:02d}_complete.json'; write(path,record)
            result['artifacts'][str(path)] = sha(path); result['retired_call_count'] += 1
            print(json.dumps(dict(event='retired',call=call,rank=rank,complete_sha256=sha(path))),flush=True)
            tick()
        env = dict(vars(legacy),np=np,flops=flops,fnp=fnp,threadpool_limits=threadpool_limits,
            threadpool_info=threadpool_info,MLP=MLP,derive_estimator_seed=derive_estimator_seed,
            materialise_predictions=materialise_predictions,args=args,inputs=result['input_bindings'],rows=rows,
            ranks=(37,),result=result,module=module,estimator=None,first_bytes=None,observer=observer,
            observed_predict=lambda estimator,mlp,capture:capture.predict(estimator,mlp),
            finish_call=finish_call,retire_call=retire_call,cold_create=cold_create)
        exec(compile(loop,'<v29-exact-paid-loop-with-owned-setup-and-capture>','exec'),env)
        require(len(result['trials']) == len(result['capture_records']) == result['retired_call_count'] == 2,
                'Exactly two completed fixed predictions')
        require(not hasattr(env['estimator'],'_v29_temporal_sink') and observer.ctx is None, 'No retained context attachment')
        estimator_ref = weakref.ref(env['estimator'])
        del env,observer,module
        gc.collect()
        require(estimator_ref() is None, 'Full numerical estimator and persistent pools retired')
        result['cold_minus_repeat_flops'] = result['trials'][0]['flops']-result['trials'][1]['flops']
        result['feature_flops'] = [c['feature_flops'] for c in result['capture_records']]
        bind(result['source_bindings'],tick); bind(result['input_bindings'],tick); bind(result['artifacts'],tick)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            all16_raw_domains_qualified=True,all16_features_qualified=True,feature_columns=30,
            repeat_byte_exact=True,feature_repeat_byte_exact=True,own_complete_bills=True,
            input_preparation_in_participant_bill=False,estimator_retired=True,setup_owned=True)
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
    print(json.dumps(dict(passed=ok,completed=ok,full_model_calls=result['full_model_calls'],error=result.get('error'))),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
