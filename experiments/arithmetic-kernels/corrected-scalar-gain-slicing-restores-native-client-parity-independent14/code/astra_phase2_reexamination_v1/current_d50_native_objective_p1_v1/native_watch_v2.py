"""Inert, reversible endpoint2 admission adapter to the frozen P1 watcher.

Only stage/source/receipt routing changes. The process-group supervision,
complete-bill progress, resource limits and failure cleanup remain literal.
No numerical imports, candidate/cache changes, or automatic next stage.
"""
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ORIGINAL = _HERE/'native_watch.py'
_ORIGINAL_SHA = '234db460741f6c42033e059a72f3a0e2c3100d3b240641fd38926bdcbcde0f2c'
_ENDPOINT_SHA = '57d0f5e464d0c2f4313336426c0c5cde0ab0d95442305c00ad5c5b9c2a7e4021'
_FAILED_NATIVE_SHA = '019f3f29df183e337b862c2bb6e540494a8cc5245cf5d51359d091d8b4d5406e'
_FAILED_WATCH_SHA = '46082a7bc015e496291cfc173c608f421c44414721d748079ba0d76ef1eea231'


def adapted_source():
    """Authenticate and invert every source-only routing edit before use."""
    original_bytes = _ORIGINAL.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != _ORIGINAL_SHA:
        raise RuntimeError('Frozen endpoint1 watcher source changed')
    original = original_bytes.decode('utf-8')
    edits = [
        ('One explicit P1 null, native40, or actual-objective score supervision.',
         'One explicit P1 matched bank/null, native40, or objective supervision.', 1),
        ("WATCH_SCHEMA = 'current_d50_native_objective_p1_native_watch_v1'",
         "WATCH_SCHEMA = 'current_d50_native_objective_p1_native_watch_v2'\n"
         f"ORIGINAL_WATCH_SHA = '{_ORIGINAL_SHA}'\n"
         f"ENDPOINT_SHA = '{_ENDPOINT_SHA}'", 1),
        ("NATIVE_SCHEMA = 'current_d50_native_objective_p1_native_v1'",
         "NATIVE_SCHEMA = 'current_d50_native_objective_p1_native_v2'", 1),
        ("SCORE_SCHEMA = 'current_d50_native_objective_p1_score_v1'",
         "SCORE_SCHEMA = 'current_d50_native_objective_p1_score_v2'", 1),
        ("ranks = [VALID[0],VALID[0]] if role == 'null' else list(VALID)+[VALID[0]]",
         "ranks = [VALID[0],VALID[0]] if role in ('bank','null') else list(VALID)+[VALID[0]]", 1),
        ("candidate_sha = BANK_SHA if role == 'null' else build['candidate_sha256']",
         "candidate_sha = BANK_SHA if role in ('bank','null') else build['candidate_sha256']", 1),
        ("'native.py'", "'native_v2.py'", 4),
        ("'judge.py'", "'judge_v2.py'", 1),
        ("'Same native source/protocol')",
         "'Same native source/protocol')\n"
         "    require(value['source_bindings'][str(HERE/'ENDPOINT_V2.md')] == ENDPOINT_SHA,\n"
         "            'Same prospective matched-history endpoint')", 1),
        ("choices=('null','native','score')", "choices=('bank','null','native','score')", 1),
        ("child_out = HERE/('score_v1' if scoring else ('native_null_v1' if args.stage == 'null' else 'native_terminal_v1'))",
         "child_out = HERE/('score2_v1' if scoring else 'native2_'+(args.stage if args.stage in ('bank','null') else 'terminal')+'_v1')", 1),
        ("out = HERE/('score_watch_v1' if scoring else ('native_null_watch_v1' if args.stage == 'null' else 'native_terminal_watch_v1'))",
         "out = HERE/('score2_watch_v1' if scoring else 'native2_'+(args.stage if args.stage in ('bank','null') else 'terminal')+'_watch_v1')", 1),
        ("    inputs = {}; bind(sources,time_guard)\n",
         "    sources[str(HERE/'native_watch.py')] = ORIGINAL_WATCH_SHA\n"
         "    sources[str(HERE/'ENDPOINT_V2.md')] = ENDPOINT_SHA\n"
         "    inputs = {\n"
         f"        str(HERE/'native_null_v1/result.json'):'{_FAILED_NATIVE_SHA}',\n"
         f"        str(HERE/'native_null_watch_v1/result.json'):'{_FAILED_WATCH_SHA}'}}\n"
         "    bind(sources,time_guard); bind(inputs,time_guard)\n"
         "    require(all(read(path).get('passed') is False for path in inputs),\n"
         "            'Preserved endpoint1 failures remain failed')\n", 1),
        ("prerequisites = dict(build_sha256=args.build_sha256)",
         "prerequisites = dict(build_sha256=args.build_sha256,endpoint_sha256=ENDPOINT_SHA)", 1),
        ("if args.stage in ('native','score'):", "if args.stage in ('null','native','score'):", 1),
        ("role = 'null' if args.stage == 'native' else 'terminal'",
         "role = {'null':'bank','native':'null','score':'terminal'}[args.stage]", 1),
        ("prior = HERE/f'native_{role}_v1'; prior_watch = HERE/f'native_{role}_watch_v1'",
         "prior = HERE/f'native2_{role}_v1'; prior_watch = HERE/f'native2_{role}_watch_v1'", 1),
        ("digest = sha(prior/'result.json',time_guard) if role == 'null' else args.native_result_sha256",
         "digest = sha(prior/'result.json',time_guard) if role in ('bank','null') else args.native_result_sha256", 1),
        ("watch_digest = sha(prior_watch/'result.json',time_guard) if role == 'null' else args.native_watch_sha256",
         "watch_digest = sha(prior_watch/'result.json',time_guard) if role in ('bank','null') else args.native_watch_sha256", 1),
        ("watch['stage'] == ('null' if role == 'null' else 'native')",
         "watch['stage'] == (role if role in ('bank','null') else 'native')", 1),
        ("watch['completed_native_bill_count'] == (2 if role == 'null' else 41)",
         "watch['completed_native_bill_count'] == (2 if role in ('bank','null') else 41)", 1),
        ("command += ['--role','null' if args.stage == 'null' else 'terminal']",
         "command += ['--role',args.stage if args.stage in ('bank','null') else 'terminal']", 1),
        ("ranks = [] if scoring else ([37,37] if args.stage == 'null' else list(VALID)+[37])",
         "ranks = [] if scoring else ([37,37] if args.stage in ('bank','null') else list(VALID)+[37])", 1),
        ("        command=command,truth_arrays_opened=False,numerical_imports=False,auto_launch_next_stage=False,",
         "        original_watch_sha256=ORIGINAL_WATCH_SHA,watch_source_inverse_verified=True,\n"
         "        endpoint_sha256=ENDPOINT_SHA,\n"
         "        command=command,truth_arrays_opened=False,numerical_imports=False,auto_launch_next_stage=False,", 1),
        ("                'null' if args.stage == 'null' else 'terminal',build,args,time_guard)",
         "                args.stage if args.stage in ('bank','null') else 'terminal',build,args,time_guard)", 1),
    ]
    changed = original
    for old,new,count in edits:
        if changed.count(old) != count:
            raise RuntimeError('Unique endpoint2 watcher source splice: '+old)
        changed = changed.replace(old,new)
    inverse = changed
    for old,new,count in reversed(edits):
        if inverse.count(new) != count:
            raise RuntimeError('Unique endpoint2 watcher inverse splice: '+new)
        inverse = inverse.replace(new,old)
    if inverse != original:
        raise RuntimeError('Exact inverse to frozen endpoint1 watcher failed')
    return changed


# Importing exposes the original inert verification API for judge_v2. The
# compiled module name is not __main__, so no stage is launched by this exec.
_namespace = dict(__name__='p1_native_watch_v2_bound_body',__file__=str(Path(__file__).resolve()))
exec(compile(adapted_source(),str(__file__)+':reversible_endpoint2','exec'),_namespace)
globals().update({key:value for key,value in _namespace.items() if not key.startswith('__')})


if __name__ == '__main__':
    raise SystemExit(main())
