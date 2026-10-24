"""DRAFT: one physical distinct-cubic VARIANCE pilot, never the old tangent law.

The complete frozen NOISE worker is source-inverted. Only the new schema,
paths, hard budget and scalar result verifier change.
Its acknowledged-context/resource/failure-ledger lifecycle remains unchanged.
No stage, FNP import, model, random draw or target access occurs at import.
"""
import ast
import hashlib
from pathlib import Path

_ADAPTER_HERE=Path(__file__).resolve().parent
_NOISE=_ADAPTER_HERE.parent/'current_comp_cubic_innovation_noise_v1'
_NOISE_SOURCES={
    _NOISE/'worker.py':'decd61b7a5fc9107e14a0b996933de0ee8c19dbd0b96f9af3c91c7059fd6c76c',
    _NOISE/'client.py':'01ace9b814a052c68a260ccc111232bfef4ea02996d2e767556641f95e249a3a',
    _NOISE/'watch.py':'12dee2bcbcc97ddb9382f6b31287309252a0d264ccd3bc14f3e351583c2d25a5',
    _NOISE/'PROTOCOL.md':'2292850f2171791feb6973dbaafa87da3418d45d78365b43a65530a038577d12',
    _NOISE/'WATCH_SPEC.md':'cb86135b168c08ff52dff63959b401a0c1b220d8f9db24f44878f1abff599835',
}
_DESIGN_SHA='97bede90799cff14b0a13fa157caf2eae0f3e34da3f431933db173e978d6e237'


def _adapt_span(source,name):
    nodes=[n for n in ast.parse(source).body
           if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1:
        raise RuntimeError('One original NOISE function: '+name)
    node=nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


_VERIFY='''def verify_call(record,mode,index,tick=None):
    require(record['schema']==SCHEMA and mode==record['mode']=='remote' and index==record['call']==0
        and all(record.get(k) is True for k in ('passed','completed','context_retired',
            'all_array_weakrefs_retired','own_complete_bill','fixed_distinct_variance_only',
            'remote_sections_are_actual_budget_deltas'))
        and record['model_calls']==0 and record['target_accessed'] is False,
        'ONE acknowledged physical variance-only RPC context')
    sections=record['sections']
    require(len(sections)==len(_KERNEL.SECTIONS)
        and [s['name'] for s in sections]==list(_KERNEL.SECTIONS)
        and all(s.get('function_completed') and s.get('tariff_matched')
            and type(s['flops']) is int and 0<=s['flops']<=cap
            and s['maximum_flops']==cap and s['expected_flops'] is None
            for s,cap in zip(sections,_KERNEL.SECTION_CAPS))
        and sum(s['flops'] for s in sections)==record['complete_flops']==record['bill']['flops_used']
        and 0<record['complete_flops']<CALL_BUDGET,
        'Every ingress, physical batch, statistic and export in the complete actual bill')
    require(len(record['rows'])==1 and record['rejected_cases']==[],
        'Only the prospective two-stream physical law')
    row=record['rows'][0];p=row['primary']
    require(row['name']=='fixed_physical_distinct_cubic_variance'
        and (row['streams'],row['batches_per_stream'],row['rows_per_batch'])==(2,16,1024)
        and row['physical_samples']==32768 and row['seeds']==list(_KERNEL.SEEDS)
        and row['matmul_counts']==dict(physical=288,query=64,vector=32)
        and row['dense_matmul_flops']==352*_KERNEL.DENSE_ONE
        and row['vector_matmul_flops']==32*_KERNEL.VECTOR_ONE
        and row['physical_prefix_forward_batches']==32 and row['hard_pathwise_ReLU_layers']==9
        and all(row[k] is True for k in ('actual_law_is_unmasked',
            'batch_k_statistic_unbiased_in_exact_arithmetic','variance_measured'))
        and all(row[k] is False for k in ('tangents_used',
            'signed_current_tensor_error_measured','raw_mean_or_final_score_measured')),
        'Actual unmasked physical law and complete fixed product inventory')
    tiny=row['tiny_identity']
    require(tiny['passed'] is True and tiny['threshold']==1e-12
        and (tiny['dimension'],tiny['rows'],tiny['columns'],tiny['direct_distinct_terms'])==(4,8,3,24)
        and all(math.isfinite(tiny[k]) and 0<=tiny[k]<=1e-12
                for k in ('maxabs_direct','maxabs_translation')),
        'Fixed independent ordered-distinct and translation identities')
    full=p['stream_trace_covariance_per_unit'];half=p['half_trace_covariance_per_unit']
    require(len(full)==2 and len(half)==4
        and all(math.isfinite(x) and x>=0 for x in full+half)
        and p['host_MSE_variance_target']==1e-9 and p['practical_sample_ceiling']==262144
        and p['stream_ratio_limit']==2. and p['half_ratio_limit']==4.
        and p['confidence_bound_claimed'] is False and p['projected_precision_not_measured'] is True,
        'Only the fixed empirical planning summaries, not a confidence bound')
    require(p['planning_multiplier']==2 and p['full_scope_projection_decision_pending'] is True,
        'Fixed empirical multiplier; whole-time practical decision still pending')
    tplan=2*max(full+half)
    expected_samples=tplan*1024/1e-9
    def same_scalar(actual,expected):
        return (type(actual) in (int,float) and math.isfinite(actual) and actual>=0
            and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-300))
    require(same_scalar(p['pooled_host_MSE_variance'],(full[0]+full[1])/64)
        and same_scalar(p['projected_noise_at_sample_ceiling'],tplan*1024/262144)
        and same_scalar(p['projected_required_samples'],expected_samples),
        'Stdlib replay of detached fixed variance/planning metadata, no target arithmetic')
    blocks=p['projected_required_pilot_blocks']
    require(type(blocks) is int and blocks==max(1,math.ceil(p['projected_required_samples']/32768)),
        'Whole fixed pilot blocks, never fractional context/time credit')
    positive=min(full+half)>0
    stream_ok=max(full)<=2*min(full);half_ok=max(half)<=4*min(half)
    require(p['positive_variance'] is positive
        and p['stream_consistency_passed'] is stream_ok
        and p['half_consistency_passed'] is half_ok
        and p['variance_planning_passed'] is (positive and stream_ok and half_ok and blocks<=8),
        'All prospective empirical variance gates, without a confidence or practical-time claim')
    require(set(row['exports'])=={'stream0','stream1'},'Exactly two detached stream checksums')
    for item in row['exports'].values():
        require(item['shape']==[16,1024] and item['dtype']=='float64' and item['bytes']==131072
            and re.fullmatch('[0-9a-f]{64}',item['raw_sha256']),
            'Paid complete FP64 stream snapshot representation')
    bill=record['bill']
    require(all(math.isfinite(bill[k]) and bill[k]>=0 for k in
        ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s'))
        and bill['wall_time_s']<60 and bill['residual_wall_time_s']<.4,
        'Original actual complete RPC wall60/residual.4 gates')'''

_COMPARE='''def compare(remote,tick=None):
    require(all(remote.get(k) is True for k in ('passed','completed','own_complete_bill',
        'one_context_completed','source_hashes_unchanged','input_hashes_unchanged'))
        and remote['retired_contexts']==1 and remote['input_bindings']==_ADMISSION['input_bindings'],
        'Fresh positive actual context and complete fixed producer inventory')
    record=call_records(remote,tick)[0]
    require(remote['calls_complete_flops']==record['complete_flops'],
        'Reported total is the actual whole acknowledged bill')
    row=record['rows'][0]
    return dict(passed=True,one_context_completed=True,primary=row['primary'],
        variance_planning_passed=row['primary']['variance_planning_passed'],
        full_scope_projection_decision_pending=True,
        complete_flops=remote['calls_complete_flops'],fixed_distinct_variance_only=True,
        model_calls=0,truth_accessed=False,variance_measured=True,
        signed_current_tensor_error_measured=False,raw_value_measured=False,
        full_estimator_fidelity_established=False,automatic_next_stage=False)'''


def source_and_proof():
    for path,digest in _NOISE_SOURCES.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen NOISE lifecycle source: '+str(path))
    original=(_NOISE/'worker.py').read_text();source=original;changes=[]
    def change(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact complete worker splice: '+before[:90])
        for unused in range(count):
            at=source.index(before);changes.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    change(_adapt_span(source,'verify_call'),_VERIFY)
    change(_adapt_span(source,'compare'),_COMPARE)
    for before,after in (
        ("SCHEMA='current_comp_cubic_innovation_noise_v1'",
         "SCHEMA='current_comp_distinct_cubic_variance_v1'"),
        ('CALL_BUDGET=14_000_000_000','CALL_BUDGET=2_000_000_000_000'),
        ('\ndef _merge(target,items,tick):',
         '\nPINS.update(_DISTINCT_EXTRA_PINS)\n\ndef _merge(target,items,tick):'),
        ("'fixed_cubic_noise_math'","'fixed_physical_distinct_cubic_math'"),
        ('cost_certificate=_KERNEL.cost_certificate(),new_arithmetic_not_inherited_fixture=True,',
         'cost_certificate=_KERNEL.cost_certificate(),complete_NOISE_worker_inverse=_DISTINCT_SOURCE_PROOF,\n        new_arithmetic_not_inherited_fixture=True,'),
        ('context_bill_environment_and_retirement_unchanged=True,one_RPC_context_only=True)',
         'context_bill_environment_and_retirement_unchanged=True,one_RPC_context_only=True,\n        complete_NOISE_worker_inverse=_DISTINCT_SOURCE_PROOF)'),
    ): change(before,after)
    for before,after in (('noise_v1','variance_v1'),
        ('fixed_noise_only','fixed_distinct_variance_only'),
        ('actual_noise_circuit_qualified','actual_variance_circuit_qualified'),
        ('cubic-innovation-noise','physical-distinct-cubic-variance')):
        change(before,after,source.count(before))
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact physical-variance worker positional inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Complete immutable NOISE source restored')
    return source,dict(owner_sha256=_NOISE_SOURCES[_NOISE/'worker.py'],
        entire_original_worker_inverse=True,splice_count=len(changes),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
        numerical_arithmetic_is_new=True,old_tangent_law_not_executed=True,
        original_closed_context_failure_and_retirement_lifecycle_unchanged=True,
        numerical_execution=False)


_derived,_adapter_proof=source_and_proof()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_distinct_variance_worker_owner',
    _DISTINCT_EXTRA_PINS={**_NOISE_SOURCES,_ADAPTER_HERE/'DESIGN.md':_DESIGN_SHA},
    _DISTINCT_SOURCE_PROOF=_adapter_proof)
exec(compile(_derived,str(_ADAPTER_HERE/'worker.py')+':NOISE-source-inverse','exec'),_api)
globals().update({k:v for k,v in _api.items() if not k.startswith('__')})

if __name__=='__main__':
    raise SystemExit(main())
