"""Inert P2-restart builder: original literal arithmetic plus restart provenance.

No build before the sole complete360-call terminal and both reviewed watches.
All numerical payload construction remains the frozen P2/P1 builder's code.
"""
from pathlib import Path
import common as c

P2_BUILDER = c.ORIGINAL/'build.py'
P2_BUILDER_SHA = 'dce1494da87a32876f9e2e29d06f4b28bbc807c65f06b9db33321512aa66ce9c'
MOCK_RESULT_SHA = 'c5a3f5b8a3dc99ca7e3550fb9ccb474b76968a49b96e00ed7e5606597476bf0e'
_restart_edits = [
    ('current_p1_native_objective_p2_watch_v1', c.RESTART_WATCH_SCHEMA, 2),
    ('current_p1_native_objective_p2_roster_v1', c.RESTART_ROSTER_SCHEMA, 1),
    ('current_p1_native_objective_p2_v1', c.RESTART_SCHEMA, 1),
    ("if __name__ == '__main__':\n    raise SystemExit(main())\n",
     '# Original entrypoint suppressed; the restart owns the sole explicit entrypoint.\n', 1),
]
_restart_source, RESTART_BUILDER_PROOF = c.splice_source(P2_BUILDER,P2_BUILDER_SHA,_restart_edits)
exec(compile(_restart_source,str(Path(__file__).resolve())+':unchanged_p2_builder','exec'),globals())
_p2_admission = admission
_restart_provenance = None


def admission(args):
    global _restart_provenance
    helper,record,keys,sources,inputs = _p2_admission(args)
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),
            'Restart provenance before numerical payload decoding')

    def pin(path,digest,source=False):
        path = str(Path(path).resolve())
        actual = sha(path)
        require(digest is None or actual == digest,'Exact restart builder provenance')
        _put(sources if source else inputs,path,actual)
        return actual

    pin(P2_BUILDER,P2_BUILDER_SHA,True)
    prefix = c.prefix_provenance(pin)
    mock = c.mock_admission(MOCK_RESULT_SHA,pin)
    qual = sealed(HERE/'qualification_v1',args.qualification_sha256)
    qw = sealed(HERE/'qualification_watch_v1',args.qualification_watch_sha256)
    tw = sealed(HERE/'training_watch_v1',args.training_watch_sha256)
    for value in (qual,qw,record,tw):
        require(value['failed_training_watch_sha256'] == c.FAILED_WATCH_SHA
                and value['old_operational_failure_still_failed'] and value['fresh_full_restart']
                and value['prefix199_authenticated_before_targets']
                and value['monitor_certificate'] == mock,
                'Every new stage uses the reviewed mock and preserved failed prefix')
    for value,count in ((qual,0),(qw,0),(record,199),(tw,199)):
        require(value['prefix199_exact_matches'] == count,'Exact complete restart-prefix count')
    for value in (qual,record):
        require(value['restart_common_inverse'] == c.RESTART_COMMON_PROOF
                and value['restart_worker_inverse']['parent_source_sha256'] == c.ORIGINAL_WORKER_SHA
                and value['restart_worker_inverse']['inverse_source_byte_exact']
                and value['partial_payload_or_optimizer_resumed'] is False
                and value['prefix199_required'] == 199,
                'Exact unchanged numerical owner and no partial optimizer/payload')
    require(record['prefix199_byte_exact'] is True and qual['prefix199_byte_exact'] is False,
            'Qualification is not a training-prefix replay')
    for value in (qw,tw):
        intervals = value['completed_sample_intervals_s']
        require(value['observational_commands_retired']
                and value['actual_sample_interval_includes_logging']
                and not value.get('observational_cleanup_error')
                and not value.get('cleanup_error') and not value.get('final_sealing_error')
                and len(intervals) > 0 and all(0 <= x <= 1 for x in intervals)
                and max(intervals) == value['maximum_sample_interval_s']
                and value['restart_watch_version_inverse']['parent_source_sha256'] == c.ORIGINAL_WATCH_SHA
                and value['restart_watch_version_inverse']['inverse_source_byte_exact']
                and value['monitor_source_inverse']['inverse_source_byte_exact']
                and value['monitor_source_inverse']['numerical_worker_edited'] is False
                and value['monitor_source_inverse']['resource_thresholds_changed'] is False,
                'Actual reviewed absolute-deadline monitor and all owned groups retired')
    require(record['training_order'] == prefix['old_training_order']
            and record['training_orders'] == prefix['old_train_orders'],
            'Exactly the interrupted fixed training trajectory, fully restarted')
    for index,expected in enumerate(prefix['prefix_signatures']):
        path = HERE/'training_v1'/f'graph_{index:04d}_complete.json'
        require(sha(path) == record['artifacts'][str(path)]
                and c.signature(read(path)) == expected,'Independent recheck of every old199 numerical signature')
    _restart_provenance = dict(restart_builder_inverse=RESTART_BUILDER_PROOF,
        failed_training_watch_sha256=c.FAILED_WATCH_SHA,
        old_operational_failure_still_failed=True,prefix199_byte_exact=True,
        prefix199_exact_matches=199,monitor_certificate=mock,
        observational_commands_retired=True,actual_sample_interval_includes_logging=True,
        partial_payload_or_optimizer_resumed=False)
    bind(sources); bind(inputs)
    return helper,record,keys,sources,inputs


# Add result metadata only. The derived numerical main is restored exactly
# by the inverse before compilation, including its60s/2GB guard and outputs.
_metadata_old = '    out.mkdir(exist_ok=False)\n'
_metadata_new = ('    require(_restart_provenance is not None, "Complete restart admission before output")\n'
                 '    result.update(_restart_provenance)\n'+_metadata_old)
require(_derived.count(_metadata_old) == 1,'Unique result-only provenance insertion')
_restart_main = _derived.replace(_metadata_old,_metadata_new)
require(_restart_main.replace(_metadata_new,_metadata_old) == _derived,
        'Exact original payload construction inverse')
exec(compile(_restart_main,str(Path(__file__).resolve())+':result_metadata_only','exec'),globals())


if __name__ == '__main__':
    raise SystemExit(main())
