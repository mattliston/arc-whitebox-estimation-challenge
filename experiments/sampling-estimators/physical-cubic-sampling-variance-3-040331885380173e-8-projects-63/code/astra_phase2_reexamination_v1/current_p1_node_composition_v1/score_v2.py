"""Versioned acquisition admission; frozen NODE/P1 score arithmetic unchanged.

No numerical execution without --run and both complete caller-pinned seals.
The original zero and its owners stay v1; the complete terminal is v2.
"""
from pathlib import Path
from types import SimpleNamespace
import endpoint_v2 as e

ORIGINAL_SCORER_SHA = 'e1d4874068b6638c95b4b66063d0abfe2e5327a07cb0ebfef1c442105e037dfe'
ENDPOINT_HELPER_SHA = 'c6ed25eaa1a6ff2131640c5ec665fccb1886446531475bff4d2d83c8076aa652'
ADMITTER_SHA = 'e8df961a574b27fa49808ba3641fdb68e6b4c37ca18b2ca3f6af04bb65f83bd5'
NEW_NATIVE_SHA = '63b2ea4edf5fd67f74a30b7fdee1ab3201a87ca13405654fb1ad0d476f0f3e34'
NEW_WATCH_SHA = '8ffc9ae68388065d49c9e39286893f5da8f5eeabe000db625643fd340442cd28'


def acquisitions_v2(args, result, pins, h, build_context):
    require, sha, read, bind, sealed = [h[k] for k in ('require', 'sha', 'read', 'bind', 'sealed')]
    require(args.native_source_sha256 == NEW_NATIVE_SHA and args.watch_source_sha256 == NEW_WATCH_SHA
            and args.zero_sha256 == e.ZERO_SHA and args.zero_watch_sha256 == e.ZERO_WATCH_SHA,
            'Fixed new terminal owners and original qualified zero')
    zn, tn = HERE/'native_zero_v1', e.OUTPUT
    zero = sealed(zn, args.zero_sha256, pins)
    zero_watch = sealed(HERE/'native_zero_watch_v1', args.zero_watch_sha256, pins)
    terminal = sealed(tn, args.terminal_sha256, pins)
    terminal_watch = sealed(e.WATCH_OUT, args.terminal_watch_sha256, pins)
    owners = SimpleNamespace(watcher_sha256=NEW_WATCH_SHA, worker_sha256=NEW_NATIVE_SHA,
        admitter_sha256=ADMITTER_SHA, helper_sha256=ENDPOINT_HELPER_SHA,
        endpoint_sha256=e.ENDPOINT_SHA, context_sha256=terminal_watch['admission_context_sha256'],
        admission_watch_sha256=terminal_watch['admission_watch_sha256'])
    context = e.read_context(owners, owners.context_sha256)
    e.retired_admission(owners, context)
    for key in ('source_bindings', 'input_bindings'):
        bind(context[key], pins)
    for path, digest in e.owners(owners).items():
        require(result['source_bindings'].get(path) == digest, 'Every executed new owner is a score source')
    require(context['candidate_sha256'] == build_context['candidates']['terminal']['sha256']
            and context['build_sha256'] == BUILD_SHA, 'Same independently replayed literal composition')
    old = e.old()
    refs, reference_pins = old.reference('zero')
    bind(reference_pins, pins)
    old.verify_acquisition(zero, read(zn/'completion.json'), args.zero_sha256,
        'zero', e.OLD_NATIVE_SHA, e.OLD_WATCH_SHA, refs)
    e.verify_native(terminal, read(tn/'completion.json'), args.terminal_sha256, owners, context)
    for role, native, watch, digest, folder, count, limit, schema in (
        ('zero', zero, zero_watch, args.zero_sha256, zn, 3, 240, old.WATCH_SCHEMA),
        ('terminal', terminal, terminal_watch, args.terminal_sha256, tn, 41, 1900, e.WATCH_SCHEMA)):
        require(watch['schema'] == schema and watch['role'] == role
                and all(watch.get(k) is True for k in ('passed', 'completed', 'hashes_unchanged',
                    'sources_unchanged', 'whole_owned_tree_retired', 'child_execution_completed'))
                and watch['child_result_sha256'] == digest
                and watch['child_completion_sha256'] == sha(folder/'completion.json')
                and watch['completed_bill_count'] == count
                and 0 < watch['maximum_sampled_rss_bytes'] < 8_000_000_000
                and watch['maximum_swap_growth_bytes'] == 0
                and 0 < watch['maximum_sample_interval_s'] <= 1 and 0 < watch['elapsed_s'] < limit,
                'Complete independently resource-qualified role with original limits')
        for path, digest2 in ((HERE/'native.py', e.OLD_NATIVE_SHA),
                              (HERE/'watch_native.py', e.OLD_WATCH_SHA),
                              (HERE/'PROTOCOL.md', PROTOCOL_SHA), (HERE/'build.py', BUILDER_SHA)):
            require(native['source_bindings'][str(path)] == digest2
                    and watch['source_bindings'][str(path)] == digest2, 'Original role/proof owners remain bound')
        require(native['candidate_sha256'] == build_context['candidates'][role]['sha256']
                and native['truth_accessed'] is False, 'Exact immutable candidate, no acquisition targets')
    require(terminal_watch['admission_process_reaped_before_native']
            and terminal_watch['first22_output_bill_domain_byte_exact']
            and terminal_watch['first22_exact_checks'] == 22
            and terminal_watch['failed_v1_watch_sha256'] == e.FAIL_SHA
            and terminal_watch['prerequisites'] == context['prerequisites']
            and terminal['admission_context_sha256'] == owners.context_sha256,
            'Full fresh terminal, original failure preserved, exact same-history controls')
    result.update(acquisition_version=2, original_zero_source_owners_preserved=True,
        failed_v1_watch_sha256=e.FAIL_SHA, first22_output_bill_domain_byte_exact=True,
        original_score_arithmetic_inverse_verified=True,
        admission_context_sha256=owners.context_sha256,
        admission_watch_sha256=owners.admission_watch_sha256)
    return {'zero': (zn, zero), 'terminal': (tn, terminal)}


e.require(e.sha(e.HERE/'score.py') == ORIGINAL_SCORER_SHA, 'Frozen original scorer owner')
_original = (e.HERE/'score.py').read_text()
_start = _original.index('    acquisitions = {}\n')
_stop = _original.index("    baseline_folder = P1/'native2_terminal_v1'\n", _start)
_old_acquisitions = _original[_start:_stop]
_extra_sources = (
    "        str(HERE/'native.py'):e.OLD_NATIVE_SHA,str(HERE/'watch_native.py'):e.OLD_WATCH_SHA,\n"
    "        str(HERE/'native_v2.py'):args.native_source_sha256,\n"
    "        str(HERE/'watch_native_v2.py'):args.watch_source_sha256,\n"
    "        str(HERE/'endpoint_v2.py'):ENDPOINT_HELPER_SHA,\n"
    "        str(HERE/'admit_v2.py'):ADMITTER_SHA,\n"
    "        str(HERE/'ENDPOINT_MEMORY_V2.md'):e.ENDPOINT_SHA,\n"
    "        str(HERE/'score.py'):ORIGINAL_SCORER_SHA,\n")
_changes = [
    ("        str(HERE/'native.py'):args.native_source_sha256,str(HERE/'watch_native.py'):args.watch_source_sha256,\n",
     _extra_sources),
    ("    out = HERE/'score_v1'", "    out = HERE/'score_v2'"),
    ("schema='current_p1_node_composition_score_v1'", "schema='current_p1_node_composition_score_v2'"),
    (_old_acquisitions, '    acquisitions = acquisitions_v2(args,result,pins,h,context)\n'),
    ('    result.update(passed=True,completed=True,hashes_unchanged=True,integrity_passed=True,',
     '    result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,integrity_passed=True,'),
    ("if __name__=='__main__':\n    raise SystemExit(main())\n",
     '# Original entrypoint suppressed; this version owns the explicit entrypoint.\n'),
]
_derived = e.splice(_original, _changes)
exec(compile(_derived, str(Path(__file__).resolve())+':unchanged_score_arithmetic', 'exec'), globals())


if __name__ == '__main__':
    raise SystemExit(main())
