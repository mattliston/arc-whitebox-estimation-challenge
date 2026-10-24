"""DRAFT: one unchanged BOX cold37; export only its cached closed summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OWNER = NEW/'check_remote_series_v1.py'
OWNER_SHA256 = 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55'
EXPECTED_FLOPS = 407741701711
TAG = 'current_comp_box_rpc_profile37_v1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def profile_summary(summary, namespace):
    """Pure host metadata, called AFTER BudgetContext.__exit__; no FNP calls."""
    require(isinstance(summary, dict) and set(summary) == {
        'flop_budget','flops_used','flops_remaining','operations','by_namespace',
        'wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s',
        'residual_wall_time_s'}, 'Exact current authoritative summary schema')
    require(summary['flop_budget'] == 2**41
        and summary['flops_used'] == EXPECTED_FLOPS
        and summary['flops_remaining'] == 2**41-EXPECTED_FLOPS,
        'Unchanged complete cold37 bill and budget')
    operations = summary['operations']
    require(isinstance(operations, dict) and 0 < len(operations) <= 128,
        'Bounded nonempty operation-category summary')
    for name, bucket in operations.items():
        require(isinstance(name, str) and len(name) <= 128 and isinstance(bucket, dict)
            and set(bucket) == {'flop_cost','calls','flopscope_backend_time_s',
                'flopscope_overhead_time_s'}, 'Exact native operation bucket schema')
        require(type(bucket['calls']) is int and bucket['calls'] > 0
            and type(bucket['flop_cost']) is int and bucket['flop_cost'] >= 0,
            'Actual operation counts and complete charged costs')
        require(all(type(bucket[key]) in (int,float) and math.isfinite(bucket[key])
            and bucket[key] >= 0 for key in ('flopscope_backend_time_s',
                'flopscope_overhead_time_s')), 'Finite nonnegative native timings')
    total_f = sum(x['flop_cost'] for x in operations.values())
    total_calls = sum(x['calls'] for x in operations.values())
    backend = math.fsum(x['flopscope_backend_time_s'] for x in operations.values())
    overhead = math.fsum(x['flopscope_overhead_time_s'] for x in operations.values())
    require(total_f == EXPECTED_FLOPS, 'Every charged operation reconciles to own bill')
    require(set(summary['by_namespace']) == {namespace}, 'Original single context namespace')
    group = summary['by_namespace'][namespace]
    require(set(group) == {'flops_used','calls','flopscope_backend_time_s',
        'flopscope_overhead_time_s','operations'} and group['operations'] == operations
        and group['flops_used'] == total_f and group['calls'] == total_calls,
        'Single namespace and operation count/F exact reconciliation')
    for actual, expected in ((group['flopscope_backend_time_s'],backend),
            (group['flopscope_overhead_time_s'],overhead)):
        require(type(actual) in (int,float) and math.isfinite(actual) and actual >= 0
            and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-7),
            'Native namespace timing reconciliation; summation roundoff only')
    timing = {key:summary[key] for key in ('wall_time_s','flopscope_backend_time_s',
        'flopscope_overhead_time_s','residual_wall_time_s')}
    require(all(type(x) in (int,float) and math.isfinite(x) and x >= 0
        for x in timing.values()), 'Finite current RPC context timing')
    wall, compute, reported_overhead, residual = (timing[k] for k in (
        'wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s',
        'residual_wall_time_s'))
    require(math.isclose(wall,compute+reported_overhead+residual,rel_tol=1e-9,abs_tol=1e-7),
        'Original RPC wall decomposition')
    gap = compute-backend-overhead
    require(math.isfinite(gap), 'Signed unassigned compute gap is reported, not clamped')
    profile = dict(schema='current_comp_box_closed_rpc_profile_v1',
        summary=summary,operation_count=total_calls,operation_flops=total_f,
        native_operation_backend_s=backend,native_operation_overhead_s=overhead,
        rpc_compute_minus_native_operation_times_s=gap,
        rpc_reported_overhead_s=reported_overhead,participant_residual_s=residual,
        wall_reconstruction_error_s=wall-(backend+overhead+gap+reported_overhead+residual),
        operation_and_namespace_flops_exact=True,operation_and_namespace_calls_exact=True,
        closed_context_cached_summary_only=True,extra_rpc_requests=0,
        diagnostic_array_operations=0,diagnostic_flops=0,
        unchanged_namespace=True,per_dispatch_or_shape_timing_measured=False,
        rpc_compute_includes_native_fnp_wrapper=True,
        native_operation_overhead_is_not_rpc_reported_overhead=True)
    require(len(json.dumps(profile,allow_nan=False).encode()) <= 131072,
        'At most128KiB metadata; no call history or array data retained')
    return profile


def source_and_proof():
    original = OWNER.read_bytes()
    require(hashlib.sha256(original).hexdigest() == OWNER_SHA256,
        'Frozen original NumPy-free paid persistent client')
    source = original.decode('utf-8')
    edits = (
        ('    for index,rank in enumerate((35,100,500,3016500,35)):\n',
         '    for index,rank in enumerate((37,)):\n'),
        ('    summary=ctx.summary_dict()\n',
         '    summary=ctx.summary_dict(by_namespace=True)\n'),
        ('                timing={key:value for key,value in summary.items() if "time" in key})\n',
         '                timing={key:value for key,value in summary.items() if "time" in key},\n'
         '                operation_profile=_profile_summary(summary,f"{args.tag}-{index}"))\n'),
        ('    assert records[0]["output_sha256"]==records[-1]["output_sha256"], "Repeated rank 35 output changed"\n',
         '    # One selected cold37; no repeat has been requested or claimed.\n'),
        ('    repeated_output_byte_identical=True,wall_s=time.time()-started,\n',
         '    no_repeat_requested=True,full_model_calls=1,wall_s=time.time()-started,\n'),
    )
    changed = source
    for before, after in edits:
        require(changed.count(before) == 1 and after not in changed, 'Unique client source splice')
        changed = changed.replace(before,after)
    inverse = changed
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique client inverse splice')
        inverse = inverse.replace(after,before)
    require(inverse.encode('utf-8') == original, 'Exact whole original client inverse')
    return changed, dict(schema='current_comp_box_rpc_profile_client_v1',
        owner_path=str(OWNER),owner_sha256=OWNER_SHA256,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole_source_inverse_verified=True,source_splices=len(edits),ranks=[37],full_model_calls=1,
        original_input_predict_fetch_budget_and_output_body_unchanged=True,
        original_context_namespace_unchanged=True,cached_summary_after_context_exit=True,
        no_live_summary_requests=True,no_repeat_claim=True,no_estimator_source_edits=True,
        diagnostic_array_operations=0,diagnostic_flops=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--tag')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.candidate and args.tag == TAG, 'Explicit candidate and single fixed profile tag')
    source, _ = source_and_proof()
    name = '_current_comp_box_rpc_profile_body'
    require(name not in sys.modules, 'Fresh client; only one physical model call')
    runtime = types.ModuleType(name)
    runtime.__file__ = str(OWNER); runtime.__package__ = None
    runtime._profile_summary = profile_summary
    sys.modules[name] = runtime
    saved = sys.argv
    try:
        sys.argv = [str(OWNER),'--candidate',args.candidate,'--tag',args.tag]
        exec(compile(source,str(OWNER),'exec'),vars(runtime))
    finally:
        sys.argv = saved
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
