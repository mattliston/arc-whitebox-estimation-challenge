"""Exact panel metadata adapter: duplicate digest memo plus fresh final union."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_DIR = Path(__file__).resolve().parent
_PARENT = _DIR/'panel_common.py'
_PARENT_SHA = '7d9c8b2d97ac5bbb85c077da694472a52fbede2946ec2643df44e1018a6185b7'
_HASH_SHA = 'db708ef69a539f918c1ad2b0ad68ffb3a9d293a05a2bb2ddae92624e1e12b653'
_PROTOCOL_SHA = 'cc4bd6dfb06d4903ad2c54fd1a222132b5b004eac95ded498ac2765e7df69233'
_V1 = {
    'panel_common.py':_PARENT_SHA,
    'panel_admit.py':'ac38f0c27d61289c53b1278309ef99b5e9d126fb7c03f06188354fdad015b83f',
    'panel_native.py':'9f6dd0de56e3af8fc2c8c3be400963148b9fde1a414b7f888a603bb988a501b3',
    'PANEL_PROTOCOL.md':'84e5c09b79db5cb4878cd2f18a550c6b83a90a25c0ce5b645e76d954a1a7097a',
    'panel_watch.py':'56793bf236fb43ef363abbff7cf8d323a828715947a625ec01b4ecd1c469335f',
    'PANEL_WATCH_SPEC.md':'c9c3df03121c7b559784a2d3114c48712babcf2805f5922a928be6dbbc144266',
    'panel_v1/terminal_watch/result.json':'97002bea32a1353ba43002729f71a7181b6a4fa31297c8d08ed70343dba79c4c',
    'panel_v1/terminal_watch/completion.json':'383440d64f70c73196d667e10fac1fa809e06e6175df667a77eb36522bf546d3',
    'panel_v1/terminal_watch/finalization.json':'f435a6eebc980ccb88a1ba955b83f7b2457bb8c473be46978ac9886af8afb513',
}


def _require(value,message):
    if not value:
        raise RuntimeError(message)


def _adapt(source,changes):
    changed = source
    for before,after in changes:
        _require(changed.count(before) == 1,'Unique versioned metadata span: '+before[:100])
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(changes):
        _require(inverse.count(after) == 1,'Unique versioned metadata inverse: '+after[:100])
        inverse = inverse.replace(after,before)
    _require(inverse == source,'Whole original metadata source restored')
    ast.parse(changed)
    return changed


_require(hashlib.sha256(_PARENT.read_bytes()).hexdigest() == _PARENT_SHA,'Frozen v1 helper')
_require(hashlib.sha256((_DIR/'panel_hash_v2.py').read_bytes()).hexdigest() == _HASH_SHA,'Reviewed target-free digest memo')
_spec = importlib.util.spec_from_file_location('cd_panel_v2_digest_owner',_DIR/'panel_hash_v2.py')
_hash_owner = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _hash_owner
_spec.loader.exec_module(_hash_owner)
METADATA_HASHES = _hash_owner.MetadataHashes(_DIR.parent.parent)


def set_hash_guard(guard):
    _require(guard is None or callable(guard),'Stage-local metadata deadline only')
    METADATA_HASHES.guard = guard


def final_bind(mappings,tick=None):
    return METADATA_HASHES.final_bind(mappings,tick)


def require_fresh_union(record):
    _require(record.get('metadata_fresh_union_rehashed') is True
        and type(record.get('metadata_unique_files_rehashed')) is int and record['metadata_unique_files_rehashed'] > 0
        and type(record.get('metadata_unique_bytes_rehashed')) is int and record['metadata_unique_bytes_rehashed'] > 0,
        'Actual v2 producer completed its unconditional fresh unique union')


def v2_lineage(values,tick=None):
    values.update({str(_DIR/name):digest for name,digest in _V1.items()})
    values[str(_DIR/'panel_hash_v2.py')] = _HASH_SHA
    values[str(_DIR/'PANEL_PROTOCOL_V2.md')] = _PROTOCOL_SHA
    for name,digest in _V1.items():
        _require(METADATA_HASHES.sha(_DIR/name,tick) == digest,'Preserved exact v1 source/failure')
    folder = _DIR/'panel_v1/terminal_watch'
    record,end,final = (json.loads((folder/name).read_text()) for name in
        ('result.json','completion.json','finalization.json'))
    _require(record['schema'] == 'current_comp_partial_d6_independent14_watch_v1'
        and all(x['passed'] is False and x['completed'] is False for x in (record,end,final))
        and record['completed_bill_count'] == 0 and len(record['phase_records']) == 1
        and record['phase_records'][0]['phase'] == 'admission'
        and record['phase_records'][0]['whole_owned_tree_retired']
        and record['whole_owned_tree_retired'] and 60 <= record['elapsed_s'] < 61
        and 'No acquisition time left before reserved completion margin' in record['error']
        and end['result_sha256'] == final['result_sha256'] == _V1['panel_v1/terminal_watch/result.json']
        and final['completion_sha256'] == _V1['panel_v1/terminal_watch/completion.json'],
        'Source-admission timeout remains failed; no native call to rescue')


_original = _PARENT.read_text()
_source = _adapt(_original,[
    ("ROOT = HERE/'panel_v1'","ROOT = HERE/'panel_v2'"),
    ("CONTEXT_SCHEMA, ADMIT_SCHEMA = PREFIX+'context_v1',PREFIX+'admission_v1'",
     "CONTEXT_SCHEMA, ADMIT_SCHEMA = PREFIX+'context_v2',PREFIX+'admission_v2'"),
    ("NATIVE_SCHEMA, WATCH_SCHEMA = PREFIX+'native_v1',PREFIX+'watch_v1'",
     "NATIVE_SCHEMA, WATCH_SCHEMA = PREFIX+'native_v2',PREFIX+'watch_v2'"),
    ("RETIRED_SCHEMA = PREFIX+'admission_watch_v1'","RETIRED_SCHEMA = PREFIX+'admission_watch_v2'"),
    ("PROTOCOL_SHA = '84e5c09b79db5cb4878cd2f18a550c6b83a90a25c0ce5b645e76d954a1a7097a'",
     'PROTOCOL_SHA = '+repr(_PROTOCOL_SHA)),
    ("OWNER_FILES = dict(watcher='panel_watch.py',worker='panel_native.py',admitter='panel_admit.py',\n"
     "    helper='panel_common.py',protocol='PANEL_PROTOCOL.md',watch_spec='PANEL_WATCH_SPEC.md')",
     "OWNER_FILES = dict(watcher='panel_watch_v2.py',worker='panel_native_v2.py',admitter='panel_admit_v2.py',\n"
     "    helper='panel_common_v2.py',protocol='PANEL_PROTOCOL_V2.md',watch_spec='PANEL_WATCH_SPEC_V2.md')"),
    ('    bind(values,tick); return values\n','    v2_lineage(values,tick); bind(values,tick); return values\n'),
    ('    return record,end,final\n','    require_fresh_union(record)\n    return record,end,final\n'),
    ("    callbacks()['verify_acquisition'](record,end,digest,args,context)\n",
     "    callbacks()['verify_acquisition'](record,end,digest,args,context)\n    require_fresh_union(record)\n"),
])
exec(compile(_source,str(_PARENT)+':unchanged-metadata-v2','exec'),globals())

# These are private, pinned stdlib metadata module dictionaries. Original bind
# function bodies remain intact; their sha lookups are the only replaced calls.
_native_metadata = e.old()
for _namespace in (d.__dict__,e.__dict__,pc.read_context.__globals__,_native_metadata.__dict__):
    _require('sha' in _namespace and callable(_namespace['sha']),'Exact metadata hash slot')
    _namespace['sha'] = METADATA_HASHES.sha
pc.sha = METADATA_HASHES.sha
sha = METADATA_HASHES.sha
_METADATA_VERSION_PROOF = dict(parent=str(_PARENT),parent_sha256=_PARENT_SHA,
    original_source_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(),whole_source_inverse_verified=True,
    original_bind_bodies_and_all_expected_digests_retained=True,
    explicit_private_metadata_hash_slots_only=True,no_global_filesystem_or_hashlib_patch=True,
    no_candidate_or_array_operator_patch=True,unconditional_final_union_required=True)
