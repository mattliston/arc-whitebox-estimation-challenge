"""Versioned end-only owned-B storage retention trim; no native execution."""
from pathlib import Path
import hashlib
import json

D = Path(__file__).resolve().parent
parent = D.parent / 'astra_oct18_bill_v1/final_union_sidebatch_v1/candidate_union_sidebatch_HELD_v1.py'
source = parent.read_text()
parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
assert parent_sha == 'b18c5d1844a3ce40a729a4d286484e1295cf9d76e727c4703d3ab21a6dda3d95'
assert '_OWP_MAX_BYTES = 1_412_000_000' in source

fragment = '''

# End-only retained storage cap. Live admission and transform arithmetic stay unchanged.
_O18_OWP_RETAIN_BYTES = 950_000_000
_O18_OWP_RETAIN_PARENT_END = _owp_end
_O18_OWP_RETAIN_STATS = dict(calls=0, trimmed_slots=0, trimmed_bytes=0, skipped_context=0)

def _owp_end():
    # _owp_predict invokes this only after its parent prediction and all inner
    # UBC/UBX/RC/RQX finally blocks have returned. Preserve original cleanup first.
    _O18_OWP_RETAIN_PARENT_END()
    _O18_OWP_RETAIN_STATS['calls'] += 1
    # Defensive no-op on an unexpected live consumer; never evict live storage.
    if (_OWP['active'] or _UBC_RUN[0] is not None or _UBC_CONTEXT[0] is not None
            or _UBX_CTX[0] is not None or _UBX_RC[0] is not None):
        _O18_OWP_RETAIN_STATS['skipped_context'] += 1
        return
    before = _OWP['bytes']
    for index in sorted(tuple(_OWP['slots']), reverse=True):
        if _OWP['bytes'] <= _O18_OWP_RETAIN_BYTES:
            break
        _owp_evict(index)
        _O18_OWP_RETAIN_STATS['trimmed_slots'] += 1
    _O18_OWP_RETAIN_STATS['trimmed_bytes'] += before - _OWP['bytes']
'''
body = source + fragment
out = D / 'endtrim095_v1'
out.mkdir(exist_ok=False)
candidate = out / 'candidate_union_sidebatch_endtrim095_HELD_v1.py'
compile(body, str(candidate), 'exec')
receipt = dict(
    status='HELD_NATIVE_UNRUN', candidate=str(candidate),
    sha256=hashlib.sha256(body.encode()).hexdigest(), parent=str(parent),
    parent_sha256=parent_sha, retained_cap_bytes=950_000_000,
    live_admission_cap_bytes=1_412_000_000, parent_prefix_exact=True,
    change='Only _owp_end; original cleanup then reverse-ordinal _owp_evict',
    lifetime_proof=[
        '_owp_predict finally calls the global _owp_end after _OWP_PARENT_PREDICT returns.',
        'Inner UBC finally drops run cache and clears run/context; UBX and RC finally clear their caches and contexts.',
        'RQX/RQX2 finally clear retained product aliases before the OWP parent returns.',
        'Original _owp_end runs first; unexpected live consumer contexts skip trimming.',
        'Evicted pool storage is rebound on demand next prediction; all transforms already recompute every invocation.'
    ],
    numeric_owners_added=0, transform_arithmetic_changed=False, route_policy_changed=False,
    source_only=True, models=0,
    limitations=['Extra future allocation/view/reshape work and wall time require native measurement.',
                 'Released references are expected to reduce peak; physical RSS reduction is not guaranteed.'])
for path, text in ((candidate, body), (out/'fragment_v1.py', fragment),
                   (out/'BUILD_v1.json', json.dumps(receipt, indent=2)+'\n')):
    with path.open('x') as handle:
        handle.write(text)
print(json.dumps(receipt))
