"""Validate a worker result against its immutable generated truth plan."""
from otr_generated_manifest_v1 import digest, require


def validate_record(plan, record, rank):
    rank = int(rank)
    require(record.get('truth_plan') == plan.spec(), 'Result truth plan mismatch')
    if rank < 7_000_000:
        require(record.get('data_source') == 'public' and record.get('truth_metadata') is None,
                'Public result has generated truth metadata')
        return
    entry, expected_sha, completion = plan.resolve(rank)
    meta = record.get('truth_metadata') or {}
    require(record.get('data_source') == 'generated', 'Generated result source mismatch')
    expected = dict(plan=plan.spec(), seed=rank, path=entry['path'], sha256=expected_sha,
                    N=entry['N'], N_half=entry['N_half'], role=entry['role'],
                    source_sha256=plan.sources)
    for key, value in expected.items():
        require(meta.get(key) == value, 'Generated result metadata mismatch: ' + key)
    require(digest(meta.get('weights_sha256')), 'Missing result weights SHA')
    if completion:
        require(meta['weights_sha256'] == completion['weights_sha256'], 'Result/completion weights mismatch')
        receipt = meta.get('completion_receipt') or {}
        from otr_generated_manifest_v1 import sha
        require(receipt == {'path': entry['completion_receipt'], 'sha256': sha(entry['completion_receipt'])},
                'Result completion receipt mismatch')
    else:
        require(meta.get('completion_receipt') is None, 'Unexpected completion receipt')
