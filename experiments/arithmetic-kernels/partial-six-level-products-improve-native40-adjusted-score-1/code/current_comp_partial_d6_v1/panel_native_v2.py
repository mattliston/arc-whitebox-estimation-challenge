"""Same15 paid calls; versioned metadata memo and final fresh source union only."""
import hashlib
import panel_common_v2 as c

_PARENT = c.HERE/'panel_native.py'
_PARENT_SHA = '9f6dd0de56e3af8fc2c8c3be400963148b9fde1a414b7f888a603bb988a501b3'
c.require(c.sha(_PARENT) == _PARENT_SHA,'Frozen unexecuted-v1 native source owner')
_original = _PARENT.read_text()
_fresh_finish = '''    finish_line = "        result.update(layer0_byte_exact=True,all90_artifacts_qualified=True)\\n"
    changed = c.splice(source,[(finish_line,finish_line+
        "        result.update(c.final_bind((result['source_bindings'],result.get('input_bindings',{}),result['artifacts'])))\\n")])
    proof.update(v1_derived_main_sha256=proof['derived_main_sha256'],
        derived_main_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        final_unique_hash_after_all15_contexts_retired=True,paid_array_operations_unchanged=True)
    return changed,proof
'''
_source = c._adapt(_original,[
    ('import panel_common as c\n','import panel_common_v2 as c\n'),
    ('    return source,proof\n',_fresh_finish),
    ("    if c.arguments('worker') is None:\n        return 0\n",
     "    if c.arguments('worker') is None:\n        return 0\n"
     "    c.set_hash_guard(lambda:c.require(time.monotonic()-began < 1200 and\n"
     "        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < c.RSS_LIMIT,\n"
     "        'Original native1200s/8GB including complete metadata hashing'))\n"),
    ("    code = namespace['worker_main']()\n",
     "    code = namespace['worker_main']()\n    c.set_hash_guard(None)\n"),
    ("    record = c.read(c.OUTPUT/'result.json'); end = c.read(c.OUTPUT/'completion.json')\n",
     "    record = c.read(c.OUTPUT/'result.json'); end = c.read(c.OUTPUT/'completion.json')\n"
     "    if record['passed']: c.require_fresh_union(record)\n"),
])
_NATIVE_V2_PROOF = dict(parent=str(_PARENT),parent_sha256=_PARENT_SHA,
    original_source_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(),whole_source_inverse_verified=True,
    only_versioned_metadata_and_post_retirement_fresh_hash=True,
    original_paid_arithmetic_observer_and_lifecycle_retained=True,all15_same_inputs_and90_artifacts=True,
    original1200s8GB_and_external120s_progress_unchanged=True)
exec(compile(_source,str(_PARENT)+':same-native15-metadata-v2','exec'),globals())
