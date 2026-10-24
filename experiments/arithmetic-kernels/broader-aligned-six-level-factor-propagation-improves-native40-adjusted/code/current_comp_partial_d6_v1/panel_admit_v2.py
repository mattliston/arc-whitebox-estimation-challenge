"""Versioned exact source admission, memoized duplicate reads, fresh final union."""
import hashlib
from pathlib import Path
import panel_common_v2 as c

_PARENT = c.HERE/'panel_admit.py'
_PARENT_SHA = 'ac38f0c27d61289c53b1278309ef99b5e9d126fb7c03f06188354fdad015b83f'
c.require(c.sha(_PARENT) == _PARENT_SHA,'Frozen failed-v1 admission source owner')
_original = _PARENT.read_text()
_source = c._adapt(_original,[
    ('import panel_common as c\n','import panel_common_v2 as c\n'),
    ('    c.bind(sources,guard); c.bind(inputs,guard); c.environment(); guard()\n',
     '    fresh = c.final_bind((sources,inputs),guard); c.environment(); guard()\n'),
    ('        full_frozen_admission_passed=True,estimator_executed=False,numerical_imports=False,truth_accessed=False)\n',
     '        full_frozen_admission_passed=True,estimator_executed=False,numerical_imports=False,truth_accessed=False,**fresh)\n'),
    ('    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)\n',
     '    c.set_hash_guard(guard)\n'
     '    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)\n'),
    ("        result['artifacts'][str(c.ADMISSION/'context.json')] = result['context_sha256']; guard()\n",
     "        c.require_fresh_union(context)\n"
     "        result.update({k:context[k] for k in ('metadata_fresh_union_rehashed',\n"
     "            'metadata_unique_files_rehashed','metadata_unique_bytes_rehashed')})\n"
     "        result['artifacts'][str(c.ADMISSION/'context.json')] = result['context_sha256']; guard()\n"),
    ('    finally:\n        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)\n',
     '    finally:\n        c.set_hash_guard(None)  # Failed-seal publication is still elapsed/RSS checked below.\n'
     '        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)\n'),
])
_ADMISSION_V2_PROOF = dict(parent=str(_PARENT),parent_sha256=_PARENT_SHA,
    original_source_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(),whole_source_inverse_verified=True,
    all_original_eligibility_checks_retained=True,original60s2GB_bound_unchanged=True,
    unconditional_unique_final_hash_before_context_publication=True,no_numerical_or_target_import=True)
exec(compile(_source,str(_PARENT)+':same-source-admission-v2','exec'),globals())
