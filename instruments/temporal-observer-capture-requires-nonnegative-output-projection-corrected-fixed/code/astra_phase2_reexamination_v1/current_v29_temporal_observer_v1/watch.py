"""Inert source-inverse of the fixed two-call native watchdog; v29 capture only."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_probe.py'
OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def derive_source():
    original = OWNER.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == OWNER_SHA, 'Frozen two-call watchdog')
    old_owners = """    own = {HERE/'watch_probe.py':args.source_sha256,HERE/'probe.py':args.worker_sha256,
        HERE/'PROBE_PROTOCOL.md':args.protocol_sha256,HERE/'PROBE_WATCH_SPEC.md':args.spec_sha256,
        HERE/'compose.py':args.compose_sha256,HERE/'fnp.py':args.fnp_sha256,
        HERE/'PROTOCOL.md':args.design_sha256,TRANSPORT:TRANSPORT_SHA}
"""
    new_owners = """    own = {HERE/'watch.py':args.source_sha256,HERE/'worker.py':args.worker_sha256,
        HERE/'ACQUISITION_PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256,
        HERE/'adapter.py':args.adapter_sha256,HERE/'PROTOCOL.md':args.adapter_protocol_sha256,
        TRANSPORT:TRANSPORT_SHA,OWNER:OWNER_SHA}
"""
    old_compose = """        composition = load(HERE/'compose.py','shared384_passive_composition',args.compose_sha256)
        pins = composition.authenticate(tick)
        for path,digest in pins.items():
            require(sha(path,tick) == digest,'Complete composition input pins')
            result['source_bindings'][str(path)] = digest
        text,proof = composition.derive_source(tick)
"""
    new_compose = """        require(args.adapter_sha256 == probe.ADAPTER_SHA
            and args.adapter_protocol_sha256 == probe.ADAPTER_PROTOCOL_SHA,'Frozen feature adapter')
        adapter = load(HERE/'adapter.py','v29_watch_frozen_adapter',args.adapter_sha256)
        pins = adapter.authenticate()
        for path,digest in pins.items():
            require(sha(path,tick) == digest,'Complete public source/license/roster/tariff pins')
            result['source_bindings'][str(path)] = digest
        text,proof = adapter.derive_source()
        tick()
"""
    old_flags = """            and record['ranks'] == [37,37] and record['all16_domains_qualified']
            and record['all47_selectors_qualified'] and record['repeat_byte_exact'] and record['own_complete_bills']
            and record['unaffected_prefix_checks'] == [dict(call=i,layers=5,byte_exact=True) for i in range(2)]
"""
    new_flags = """            and record['schema'] == 'current_v29_temporal_acquisition_v1'
            and record['ranks'] == [37,37] and record['all16_raw_domains_qualified']
            and record['all16_features_qualified'] and record['feature_columns'] == 30
            and record['repeat_byte_exact'] and record['feature_repeat_byte_exact'] and record['own_complete_bills']
            and record['setup_calls'] == 1 and record['setup_seed'] == 0 and record['setup_owned']
            and record['estimator_retired'] and record['head_attached'] is False and record['feedback_enabled'] is False
"""
    edits = (
        ('KEYS = (\'source\',\'worker\',\'protocol\',\'spec\',\'compose\',\'fnp\',\'design\')',
         'KEYS = (\'source\',\'worker\',\'protocol\',\'spec\',\'adapter\',\'adapter-protocol\')'),
        (old_owners,new_owners),
        ("child_folder,out = HERE/'probe_v1',HERE/'probe_watch_v1'",
         "child_folder,out = HERE/'acquisition_v1',HERE/'acquisition_watch_v1'"),
        ("schema='current_comp_shared_old384_native37_watch_v1'",
         "schema='current_v29_temporal_acquisition_watch_v1'"),
        ("probe = load(HERE/'probe.py','shared384_passive_probe_api',args.worker_sha256)",
         "probe = load(HERE/'worker.py','v29_passive_worker_api',args.worker_sha256)"),
        (old_compose,new_compose),
        ("str(HERE/'probe.py'),'--run',", "str(HERE/'worker.py'),'--run',"),
        ("            '--compose-sha256',args.compose_sha256,'--fnp-sha256',args.fnp_sha256,'--design-sha256',args.design_sha256]",
         "            '--adapter-sha256',args.adapter_sha256,'--adapter-protocol-sha256',args.adapter_protocol_sha256]"),
        (old_flags,new_flags),
    )
    changed = original
    for old,new in edits:
        require(changed.count(old) == 1, 'Unique v29 watchdog substitution '+old[:70])
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique v29 watchdog inverse '+new[:70])
        inverse = inverse.replace(new,old)
    require(inverse == original, 'Whole watchdog byte inverse; no resource/cleanup loop changes')
    return changed, dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_300s_8GB_zero_swap_120s_progress_1s_actual_observations=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert. Explicit --run --numerical-slot-clear and reviewed hashes are required.')
    source, proof = derive_source()
    scope = dict(__name__='__main__',__file__=str(Path(__file__).resolve()),OWNER=OWNER,OWNER_SHA=OWNER_SHA)
    exec(compile(source,str(Path(__file__).resolve())+':qualified-watch-inverse','exec'),scope)
