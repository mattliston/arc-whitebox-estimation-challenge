"""Unchanged qualified persistent-five monitor, current APUSH identities only."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_partial_d6_v1/production_watch_v2.py'
OWNER_SHA = '5e375679330dfdc0623140aede5a5bb3851f71588387a8e65d54e6d889f2361f'


def source_and_proof():
    original = OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != OWNER_SHA:
        raise RuntimeError('Frozen qualified production-five supervisor owner')
    edits = [
        ('current_comp_partial_d6', 'current_comp_aligned_push_d6', 4),
        ('ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d',
         '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a', 1),
        ("'production_check_v2.py'", "'production_check.py'", 2),
        ("'PRODUCTION_PROTOCOL_V2.md'", "'PRODUCTION_PROTOCOL.md'", 2),
        ("'PRODUCTION_WATCH_SPEC_V2.md'", "'PRODUCTION_WATCH_SPEC.md'", 2),
        ("        TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}",
         "        TRANSPORT:TRANSPORT_SHA,**h.MON_PINS,\n"
         "        APUSH_PRODUCTION_OWNER:APUSH_PRODUCTION_OWNER_SHA}", 1),
        ("        numerical_slot_clear_attested=True)",
         "        numerical_slot_clear_attested=True,\n"
         "        supervisor_source_inverse=APUSH_PRODUCTION_SOURCE_PROOF)", 1),
    ]
    source = original
    for before, after, count in edits:
        if source.count(before) != count or after in source:
            raise RuntimeError('Exact production-five source span: '+before)
        source = source.replace(before, after)
    inverse = source
    for before, after, count in reversed(edits):
        if inverse.count(after) != count:
            raise RuntimeError('Exact production-five inverse span: '+after)
        inverse = inverse.replace(after, before)
    if inverse != original:
        raise RuntimeError('Whole qualified production-five source restored')
    ast.parse(source)
    return source, dict(owner_path=str(OWNER), owner_sha256=OWNER_SHA,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
        whole_source_inverse_verified=True, exact_original_five_inputs_and_client=True,
        original_monitor_resource_deadlines_and_cleanup=True,
        same60s_wall_point4s_residual_and8GB_limits=True,
        own_complete_initialization_1944970298F=True,
        no_numerical_import_or_candidate_change=True,
        changes=['current APUSH identities and source owners', 'explicit source-inverse provenance'])


def runtime():
    source, proof = source_and_proof()
    namespace = dict(__file__=str(Path(__file__).resolve()),
        __name__='apush_production_five_supervisor',
        APUSH_PRODUCTION_OWNER=OWNER, APUSH_PRODUCTION_OWNER_SHA=OWNER_SHA,
        APUSH_PRODUCTION_SOURCE_PROOF=proof)
    exec(compile(source, str(OWNER)+':APUSH-identities-only', 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    raise SystemExit(runtime()['main']())
