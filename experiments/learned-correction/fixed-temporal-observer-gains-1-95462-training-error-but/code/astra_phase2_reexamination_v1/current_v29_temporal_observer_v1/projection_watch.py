"""Inert exact resource-loop inverse for projected v29 cold37/repeat37 only."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V29_WATCH_OWNER = HERE/'watch.py'
V29_WATCH_SHA = 'f7783cd003fdca3bd625ffa62733fbb12655704998cb2547e00dcee736b8e0b6'
V29_WATCH_SPEC = HERE/'WATCH_SPEC.md'
V29_WATCH_SPEC_SHA = 'ae85478753619afcb62d82eb0d6e4d23290b3eafc8f4df34750f4176feda6619'
OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_probe.py'
OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def derive_source():
    require(hashlib.sha256(V29_WATCH_OWNER.read_bytes()).hexdigest() == V29_WATCH_SHA,
            'Frozen original v29 source adapter, including its failed receipts')
    spec = importlib.util.spec_from_file_location('v29_projection_original_watch',V29_WATCH_OWNER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    original,parent = value.derive_source()
    require(value.OWNER == OWNER and value.OWNER_SHA == OWNER_SHA,'Exact original resource-loop owner')
    old_owners = """    own = {HERE/'watch.py':args.source_sha256,HERE/'worker.py':args.worker_sha256,
        HERE/'ACQUISITION_PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256,
        HERE/'adapter.py':args.adapter_sha256,HERE/'PROTOCOL.md':args.adapter_protocol_sha256,
        TRANSPORT:TRANSPORT_SHA,OWNER:OWNER_SHA}
"""
    new_owners = """    require(args.protocol_sha256 == args.adapter_protocol_sha256,'One explicit projection protocol owner')
    own = {HERE/'projection_watch.py':args.source_sha256,HERE/'projection_worker.py':args.worker_sha256,
        HERE/'PROJECTION_PROTOCOL.md':args.protocol_sha256,HERE/'PROJECTION_WATCH_SPEC.md':args.spec_sha256,
        HERE/'projection_adapter.py':args.adapter_sha256,TRANSPORT:TRANSPORT_SHA,OWNER:OWNER_SHA,
        V29_WATCH_OWNER:V29_WATCH_SHA,V29_WATCH_SPEC:V29_WATCH_SPEC_SHA}
"""
    old_flags = """            and record['schema'] == 'current_v29_temporal_acquisition_v1'
            and record['ranks'] == [37,37] and record['all16_raw_domains_qualified']
            and record['all16_features_qualified'] and record['feature_columns'] == 30
            and record['repeat_byte_exact'] and record['feature_repeat_byte_exact'] and record['own_complete_bills']
            and record['setup_calls'] == 1 and record['setup_seed'] == 0 and record['setup_owned']
            and record['estimator_retired'] and record['head_attached'] is False and record['feedback_enabled'] is False
"""
    new_flags = """            and record['schema'] == 'current_v29_output_projection_qualification_v1'
            and record['ranks'] == [37,37] and record['all16_raw_domains_qualified']
            and record['all16_features_qualified'] and record['feature_columns'] == 30
            and record['repeat_byte_exact'] and record['feature_repeat_byte_exact'] and record['own_complete_bills']
            and record['setup_calls'] == 1 and record['setup_seed'] == 0 and record['setup_owned']
            and record['estimator_retired'] and record['head_attached'] is False and record['feedback_enabled'] is False
            and record['raw_hosts_preserved'] and record['all16_projection_byte_exact']
            and record['projection_tariff_qualified'] and record['projection_flops'] == [16384,16384]
            and record['original_negative_acquisition_preserved'] and record['downstream_training_authorized'] is False
            and record['parameter_updates'] == 0
            and len(record['projection_records']) == 2
            and record['original_unclipped_domain_passed'] ==
                [p['original_unclipped_domain_passed'] for p in record['projection_records']]
            and all(p['call'] == i and p['rank'] == 37 and len(p['layers']) == 16
                and p['raw_hosts_finite'] and p['projected_output_domain_passed']
                and p['served_equals_projected_raw_host_byte_exact']
                and p['projection_flops'] == 16384 and p['projection_operation_count'] == 1
                and p['projection_tariff_qualified'] and p['projection_in_complete_bill']
                and p['closed_bill_unchanged_by_statistics'] and p['statistics_after_context_exit']
                and p['own_complete_flops'] == record['trials'][i]['flops']
                and all(row['layer'] == l and row['raw_host']['nonfinite_count'] == 0
                    and row['served']['nonfinite_count'] == row['served']['negative_count'] == 0
                    and row['served_equals_projected_raw_host_byte_exact'] for l,row in enumerate(p['layers']))
                for i,p in enumerate(record['projection_records']))
"""
    edits = (
        (old_owners,new_owners),
        ("child_folder,out = HERE/'acquisition_v1',HERE/'acquisition_watch_v1'",
         "child_folder,out = HERE/'projection_v1',HERE/'projection_watch_v1'"),
        ("schema='current_v29_temporal_acquisition_watch_v1'",
         "schema='current_v29_output_projection_qualification_watch_v1'"),
        ("probe = load(HERE/'worker.py','v29_passive_worker_api',args.worker_sha256)",
         "probe = load(HERE/'projection_worker.py','v29_projection_worker_api',args.worker_sha256)"),
        ("adapter = load(HERE/'adapter.py','v29_watch_frozen_adapter',args.adapter_sha256)",
         "adapter = load(HERE/'projection_adapter.py','v29_watch_projected_adapter',args.adapter_sha256)"),
        ("str(HERE/'worker.py'),'--run',","str(HERE/'projection_worker.py'),'--run',"),
        (old_flags,new_flags),
    )
    changed = original
    for old,new in edits:
        require(changed.count(old) == 1,'Unique projection-watch substitution '+old[:70])
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1,'Unique projection-watch inverse '+new[:70])
        inverse = inverse.replace(new,old)
    require(inverse == original,'Whole source inverse, resource/deadline/cleanup loop unchanged')
    return changed,dict(parent=parent,v29_owner_sha256=V29_WATCH_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        unchanged_300s_8GB_zero_swap_120s_progress_1s_actual_observations=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert. Explicit --run --numerical-slot-clear and all reviewed SHA arguments required.')
    source, proof = derive_source()
    scope = dict(__name__='__main__',__file__=str(Path(__file__).resolve()),OWNER=OWNER,OWNER_SHA=OWNER_SHA,
        V29_WATCH_OWNER=V29_WATCH_OWNER,V29_WATCH_SHA=V29_WATCH_SHA,
        V29_WATCH_SPEC=V29_WATCH_SPEC,V29_WATCH_SPEC_SHA=V29_WATCH_SPEC_SHA)
    exec(compile(source,str(Path(__file__).resolve())+':original-resource-loop-inverse','exec'),scope)
