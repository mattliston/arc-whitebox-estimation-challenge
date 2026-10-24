"""Inert additive cold37/repeat37 output-projection qualification adapter."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OWNER = HERE/'worker.py'
OWNER_SHA = '551644da1dd6a0d7fe193df03894e889e8e6589bb839c65da64bf2917acb271a'
ADAPTER_SHA = '3bcefbe9503e8457cc8be2d9dcf3d23effd5b0217bd44dbe9c3a3fd955d29cf0'
ADAPTER_PROTOCOL_SHA = '8730fc7b1fc11bcd8f9571667710034cca0251da9ba992caa8263d948ba7871d'
PROJECTION_SOURCE_PINS = {
    OWNER:OWNER_SHA,
    HERE/'watch.py':'f7783cd003fdca3bd625ffa62733fbb12655704998cb2547e00dcee736b8e0b6',
    HERE/'ACQUISITION_PROTOCOL.md':'4ec87b4e7ce8e8a3859cb772ef16a131886912b45b577417414dce1e8261f7b5',
    HERE/'WATCH_SPEC.md':'ae85478753619afcb62d82eb0d6e4d23290b3eafc8f4df34750f4176feda6619',
    HERE.parent.parent/'data_gate/.venv/lib/python3.11/site-packages/flopscope/_budget.py':
        'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
}
NEGATIVE_PINS = {
    str(HERE/'acquisition_v1/result.json'):'ea17d5505acdcbb4e2bd2f258a28530b2cf988551f4e71e43e019e7f4ddaed49',
    str(HERE/'acquisition_v1/completion.json'):'933f69c3425c3487ffd84e2561db42682edb7141e858bde6d45889a61e1047b7',
    str(HERE/'acquisition_v1/finalization.json'):'f0ce9cc9d37f0e7a5faac90e96ed2f40dde44d7b1fa85697590a968a1cab00d4',
    str(HERE/'acquisition_watch_v1/result.json'):'6073323d59c56bd48f910acd9c640156ef0a81e036de2014e57d7f83167ac781',
    str(HERE/'acquisition_watch_v1/completion.json'):'a920c31d29a9238e0559ebab6aa8384cd6ff2fa03616975c7804dd3a025cc53e',
    str(HERE/'acquisition_watch_v1/finalization.json'):'429a6c39e3e51495e2684fc328de9268d53e5e5e8174b318e4c9fff500ab1dfd',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def environment():
    require(hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA,'Frozen original worker')
    spec = importlib.util.spec_from_file_location('v29_projection_environment_owner',OWNER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value.environment()


def projection_observe(np, observer, values, ctx, trial, result, folder):
    """Runs only after context exit; detached bytes, scalar statistics and bill metadata."""
    def closed():
        return (int(ctx.flops_used),len(ctx.op_log),ctx.wall_time_s,
                ctx.flopscope_backend_time_s,ctx.flopscope_overhead_time_s,ctx.residual_wall_time_s)
    before = closed()
    require(observer.ctx is None and len(observer.features) == 16 and sys.getprofile() is None,
            'All raw packets captured and original profiler detached after paid context')
    records = [item for item in ctx.op_log if item.op_name == 'maximum'
               and tuple(tuple(s) for s in item.shapes) == ((16,1024),())]
    tariff = [dict(op_name=item.op_name,shapes=[list(s) for s in item.shapes],
        resolved_dtype=item.resolved_dtype,flops=int(item.flop_cost),cumulative=int(item.cumulative))
        for item in records]
    def stats(value):
        finite = np.isfinite(value)
        selected = value[finite]
        negative = finite & (value < 0)
        indices = np.flatnonzero(negative)
        return dict(finite_minimum=float(selected.min()) if selected.size else None,
            finite_maximum=float(selected.max()) if selected.size else None,
            negative_count=int(negative.sum()),nan_count=int(np.isnan(value).sum()),
            positive_infinity_count=int(np.isposinf(value).sum()),
            negative_infinity_count=int(np.isneginf(value).sum()),nonfinite_count=int((~finite).sum()),
            most_negative_index=int(indices[np.argmin(value[indices])]) if indices.size else None)
    layers = []
    for layer,packet in enumerate(observer.features):
        item = packet['host']
        host = np.frombuffer(item['data'],dtype=item['dtype']).reshape(item['shape'])
        require(packet['layer'] == layer and host.dtype == np.float32 and host.shape == (1024,),
                'Original finite-host packet schema; raw host never redefined as served')
        projected = np.maximum(host,np.float32(0))
        layers.append(dict(layer=layer,raw_host=stats(host),served=stats(values[layer]),
            raw_host_sha256=hashlib.sha256(item['data']).hexdigest(),
            served_sha256=hashlib.sha256(values[layer].tobytes()).hexdigest(),
            projected_raw_host_sha256=hashlib.sha256(projected.tobytes()).hexdigest(),
            served_equals_projected_raw_host_byte_exact=projected.tobytes() == values[layer].tobytes()))
    raw_finite = all(x['raw_host']['nonfinite_count'] == 0 for x in layers)
    output_domain = all(x['served']['nonfinite_count'] == x['served']['negative_count'] == 0 for x in layers)
    identity = all(x['served_equals_projected_raw_host_byte_exact'] for x in layers)
    actual_f = sum(x['flops'] for x in tariff)
    qualified_tariff = bool(len(tariff) == 1 and actual_f == 16384 and tariff[0]['resolved_dtype'] == 'float32'
                           and 0 < tariff[0]['cumulative'] <= trial['flops'])
    record = dict(schema='current_v29_output_projection_statistics_v1',call=trial['call'],rank=trial['rank'],
        layers=layers,host_semantics='raw_physical_unclipped',served_semantics='final_output_maximum_zero',
        raw_hosts_finite=raw_finite,projected_output_domain_passed=output_domain,
        original_unclipped_domain_passed=raw_finite and all(x['raw_host']['negative_count'] == 0 for x in layers),
        served_equals_projected_raw_host_byte_exact=identity,projection_operations=tariff,
        projection_flops=actual_f,projection_operation_count=len(tariff),projection_tariff_qualified=qualified_tariff,
        own_complete_flops=trial['flops'],projection_in_complete_bill=True,statistics_after_context_exit=True,
        statistical_score_measured=False,truth_accessed=False)
    require(before == closed(), 'Offline diagnostics leave every closed bill counter unchanged')
    record['closed_bill_unchanged_by_statistics'] = True
    path = folder/f"call_{trial['call']:02d}_projection.json"
    write(path,record)
    result['artifacts'][str(path)] = sha(path)
    result['projection_records'].append(record)
    trial.update(projection_statistics=str(path),projection_statistics_sha256=sha(path),
        projection_flops=actual_f,projection_operation_count=len(tariff),
        raw_hosts_finite=raw_finite,raw_host_projection_byte_exact=identity,
        original_unclipped_domain_passed=record['original_unclipped_domain_passed'],
        projection_tariff_qualified=qualified_tariff)


def failure_retire(env, observer, result, folder):
    """False-path cleanup only; no new prediction, success marker or candidate reset."""
    references = {}
    if observer is not None:
        if sys.getprofile() is observer:
            sys.setprofile(None)
        require(sys.getprofile() is None,'No surviving owned profiler')
        observer.ctx = None
        observer.features.clear(); observer.variances.clear()
    if env is None:
        return references
    estimator = env.get('estimator')
    if estimator is not None:
        references['estimator'] = weakref.ref(estimator)
        if hasattr(estimator,'_v29_temporal_sink'):
            delattr(estimator,'_v29_temporal_sink')
    del estimator
    for key,start_key in (('input_ctx','input_started'),('ctx','started')):
        context = env.pop(key,None)
        if context is None:
            continue
        # The original lexical with has exited before exec raises into this handler.
        bill = folder/f"failed_call_{env.get('call',0):02d}_{key}_bill.json"
        attempted = dict(call=int(env.get('call',0)),rank=int(env.get('rank',37)),
            **env['metrics'](context,env.get(start_key,time.monotonic())),
            passed=False,completed=False,context_exited=True,diagnostic_failure_receipt=True)
        write(bill,attempted); result['artifacts'][str(bill)] = sha(bill)
        references[key] = weakref.ref(context)
        reset = env['reset_saved_history'](env['flops'],context,bill)
        reset.update(passed=False,completed=False,qualified_call=False,
            reset_after_lexical_context_exit=True,context_reference_name=key)
        path = folder/f"failed_call_{env.get('call',0):02d}_{key}_reset.json"
        write(path,reset); result['artifacts'][str(path)] = sha(path)
        del context
    env.clear()
    return references


def _selected_helpers():
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    answer = []
    for name in ('projection_observe','failure_retire'):
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name]
        require(len(nodes) == 1,'One source-owned projection helper '+name)
        n = nodes[0]
        answer.append(''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno]))
    return '\n\n'.join(answer)+'\n\n'


def derive_source():
    original = OWNER.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == OWNER_SHA,'Frozen original acquisition source')
    old_paid = """    changed = original.replace(before,after)
    require(changed.count(after) == 1 and changed.replace(after,before) == original,
            'Whole paid-loop inverse, sole explicit setup addition')
    return changed, dict(parent=prior, setup_splice_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_owned_inputs_bills_materialization_resets_release_and_repeat=True)
"""
    new_paid = """    changed = original.replace(before,after)
    gate = '        assert np.isfinite(values).all() and (values>=0).all()'
    observed = ('        projection_observe(np,observer,values,ctx,trial,result,args.out)\\n'
        + gate + '\\n        require(trial[\\'raw_hosts_finite\\'] and trial[\\'raw_host_projection_byte_exact\\']'
        + ' and trial[\\'projection_tariff_qualified\\'],\\'Exact raw-host projection and own tariff\\')')
    require(changed.count(gate) == 1,'Unique original output-domain gate')
    changed = changed.replace(gate,observed)
    inverse = changed.replace(observed,gate).replace(after,before)
    require(changed.count(observed) == 1 and inverse == original,
            'Whole paid-loop inverse: setup and post-exit diagnostic only')
    return changed, dict(parent=prior,setup_splice_inverse=True,post_context_statistics_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_owned_inputs_bills_materialization_resets_release_and_repeat=True,
        original_strict_served_domain_assertion_unchanged=True)
"""
    old_host = """                require(arrays[f'layer_{layer:02d}_host'].tobytes() == served[layer].tobytes(),
                        'Head-free captured host equals actual served row')
"""
    new_host = """                require(np.maximum(arrays[f'layer_{layer:02d}_host'],np.float32(0)).tobytes()
                        == served[layer].tobytes(), 'Raw physical host projects to actual served row exactly')
"""
    old_except = """    except BaseException:
        result['error'] = traceback.format_exc()
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
"""
    new_except = """    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        if not result['passed']:
            try:
                references = failure_retire(env,observer,result,args.out)
                env=observer=module=None
                gc.collect()
                require(all(ref() is None for ref in references.values()),'Failed contexts/estimator actually retired')
                result.update(failure_cleanup_passed=True,failed_owner_references_retired=list(references))
            except BaseException:
                result.update(passed=False,completed=False,failure_cleanup_passed=False,
                              cleanup_error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
"""
    edits = [
        ("ADAPTER_SHA = 'cab844cd7981d31c7e44f4797ad60490729bdfadb8ce46655cdd5966ae9f36b7'",f"ADAPTER_SHA = '{ADAPTER_SHA}'",1),
        ("ADAPTER_PROTOCOL_SHA = '2a93907e69427a55c9a11eef7bd2143d9f71835d62f5eb4323700c1a8bd545ba'",f"ADAPTER_PROTOCOL_SHA = '{ADAPTER_PROTOCOL_SHA}'",1),
        ("HERE/'worker.py'","HERE/'projection_worker.py'",1),
        ("HERE/'watch.py'","HERE/'projection_watch.py'",1),
        ("HERE/'adapter.py'","HERE/'projection_adapter.py'",2),
        ("HERE/'PROTOCOL.md': ADAPTER_PROTOCOL_SHA","HERE/'PROJECTION_PROTOCOL.md': ADAPTER_PROTOCOL_SHA",1),
        ("HERE/'ACQUISITION_PROTOCOL.md':args.protocol_sha256","HERE/'PROJECTION_PROTOCOL.md':args.protocol_sha256",1),
        ("HERE/'WATCH_SPEC.md'","HERE/'PROJECTION_WATCH_SPEC.md'",1),
        ('require(args.adapter_sha256 == ADAPTER_SHA and args.adapter_protocol_sha256 == ADAPTER_PROTOCOL_SHA,',
         'require(args.adapter_sha256 == ADAPTER_SHA and args.adapter_protocol_sha256 == ADAPTER_PROTOCOL_SHA\n'
         '            and args.protocol_sha256 == args.adapter_protocol_sha256,',1),
        ("CAP, RSS, BUDGET = 300., 8_000_000_000, 2**41",
         "SOURCE_PINS.update(PROJECTION_SOURCE_PINS)\nCAP, RSS, BUDGET = 300., 8_000_000_000, 2**41",1),
        (old_paid,new_paid,1),
        ('def main():',_selected_helpers()+'def main():',1),
        ("args.out = HERE/'acquisition_v1'","args.out = HERE/'projection_v1'",1),
        ("schema='current_v29_temporal_acquisition_v1'","schema='current_v29_output_projection_qualification_v1'",1),
        ("        capture_records=[],full_model_calls=0,retired_call_count=0,parameter_updates=0,",
         "        capture_records=[],projection_records=[],full_model_calls=0,retired_call_count=0,parameter_updates=0,",1),
        ("    try:\n        bind(result['source_bindings'],tick)",
         "    env=observer=module=None\n    try:\n        bind(result['source_bindings'],tick)\n"
         "        bind(NEGATIVE_PINS,tick)\n"
         "        require(all(json.loads(Path(p).read_text()).get('passed') is False\n"
         "            and json.loads(Path(p).read_text()).get('completed') is False for p in NEGATIVE_PINS),\n"
         "            'Original failed acquisition remains negative, never a prerequisite PASS')\n"
         "        result['input_bindings'].update(NEGATIVE_PINS)",1),
        (old_host,new_host,1),
        ('feature_F_in_actual_bill=True,host_served_byte_exact=True,all16_raw_domains=True,',
         "feature_F_in_actual_bill=True,raw_host_projected_served_byte_exact=True,\n"
         "                host_semantics='raw_physical_unclipped',all16_raw_domains=True,",1),
        ('            finish_call=finish_call,retire_call=retire_call,cold_create=cold_create)',
         '            finish_call=finish_call,retire_call=retire_call,cold_create=cold_create,\n'
         '            projection_observe=projection_observe)',1),
        ('        del env,observer,module\n        gc.collect()',
         '        env.clear(); env=observer=module=None\n        gc.collect()',1),
        ("            input_preparation_in_participant_bill=False,estimator_retired=True,setup_owned=True)",
         "            input_preparation_in_participant_bill=False,estimator_retired=True,setup_owned=True,\n"
         "            raw_hosts_preserved=True,all16_projection_byte_exact=True,projection_tariff_qualified=True,\n"
         "            projection_flops=[p['projection_flops'] for p in result['projection_records']],\n"
         "            original_unclipped_domain_passed=[p['original_unclipped_domain_passed'] for p in result['projection_records']],\n"
         "            original_negative_acquisition_preserved=True,downstream_training_authorized=False)",1),
        (old_except,new_except,1),
    ]
    changed = original
    for old,new,count in edits:
        require(changed.count(old) == count,'Exact projection worker splice '+old[:80])
        changed = changed.replace(old,new)
    inverse = changed
    for old,new,count in reversed(edits):
        require(inverse.count(new) == count,'Exact projection worker inverse '+new[:80])
        inverse = inverse.replace(new,old)
    require(inverse == original,'Whole worker source inverse; no rewritten paid numerical loop')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_strict_domain_and_300s_8GB_120s_guards_retained=True,failed_acquisition_not_rescued=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert. Use explicit --run and all reviewed SHA arguments after independent review.')
    source, proof = derive_source()
    scope = dict(__name__='__main__',__file__=str(Path(__file__).resolve()),
                 PROJECTION_SOURCE_PINS=PROJECTION_SOURCE_PINS,NEGATIVE_PINS=NEGATIVE_PINS)
    exec(compile(source,str(Path(__file__).resolve())+':original-worker-inverse','exec'),scope)
