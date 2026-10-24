"""Read public metadata and projected identities only; no moment payload path."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
GATE = HERE.parent/'data_gate'
sys.path.insert(0, str(GATE))
import target_blind_mini_v2 as mini

OUT = HERE/'public_p2_moments_metadata_v1.json'
CARD = HERE/'public_p2_moments_source_card_v1.txt'
assert not OUT.exists() and not CARD.exists()
REPO = 'keenanpepper/arc-whestbench-p2-higher-moments-2026'
PIN = '1bb00b739619154ca255db6443af0b6a9dd960ce'
requests = []


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent':'phase2-metadata-audit/1'})
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read(1000001)
    assert len(data) <= 1000000, 'metadata endpoint unexpectedly large'
    requests.append(dict(url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return data


info = json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}/revision/{PIN}'))
assert info['sha'] == PIN and info['private'] is False and not info['gated']
tree = json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}/tree/{PIN}/moments?recursive=false&expand=false&limit=1000'))
files = {entry['path']:entry for entry in tree if entry['type'] == 'file'}
assert set(files) == {f'moments/mlp_{i:05d}.npz' for i in range(100)}
card = fetch(f'https://huggingface.co/datasets/{REPO}/raw/{PIN}/README.md')
assert b'v2-phase2' in card and b'1024' in card and b'1e8' in card
admission = mini.validate_pinned_mini_snapshot()
identities = []
columns = ['mlp_id','mlp_name','mlp_seed']
for shard_index, entry in enumerate(mini.AUTHORITY['splits']['mini']['shards']):
    shard = mini._attest_shard_pointer(mini.pinned_snapshot_path()/entry['path'], entry)
    batch = pq.read_table(shard, columns=columns, use_threads=False)
    assert batch.column_names == columns
    for row_in_shard, row in enumerate(batch.to_pylist()):
        row.update(ordinal=len(identities), shard_index=shard_index, row_in_shard=row_in_shard)
        row['assignment_sha256'] = hashlib.sha256(('astra-p2-third-plane-pilot-v1|'+str(row['mlp_seed'])).encode()).hexdigest()
        identities.append(row)
assert len(identities) == 100 and len({row['mlp_seed'] for row in identities}) == 100
full_path = GATE/'full_network_splits.json'
full = json.loads(full_path.read_text())
full_seeds = {row['mlp_seed'] for group in full['splits'].values() for row in group}
mini_seeds = {row['mlp_seed'] for row in identities}
assert len(full_seeds) == 1000
overlap = dict(full=sorted(mini_seeds & full_seeds), virgin=sorted(mini_seeds & set(range(3016500,3016506))))
assert not any(overlap.values()), overlap
selected = sorted(identities, key=lambda row:row['assignment_sha256'])[:8]
for i, row in enumerate(selected):
    row['role'] = 'train' if i < 6 else 'heldout'
    entry = files[f"moments/mlp_{row['ordinal']:05d}.npz"]
    row['moment_file'] = dict(path=entry['path'], bytes=entry['size'], lfs_sha256=entry['lfs']['oid'])
    assert len(row['moment_file']['lfs_sha256']) == 64
record = dict(passed=True, repository=REPO, commit=PIN, last_modified=info['lastModified'],
              source_mini_commit=mini.AUTHORITY['pinned_commit_sha'], identities_decoded=100,
              decoded_columns=columns, target_values_decoded=0, moment_payload_bytes_fetched=0,
              panel_seed_overlap=overlap, full_manifest_sha256=hashlib.sha256(full_path.read_bytes()).hexdigest(),
              selected=selected, total_selected_moment_bytes=sum(row['moment_file']['bytes'] for row in selected),
              requests=requests, utc_finished=datetime.now(timezone.utc).isoformat(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              qualification='Public metadata plus local projected identities only. No moment NPZ downloaded or labels decoded. Fixed potential6/2 split, distinct from entire full1000 and virgin6 by seed. Numerical/data qualification and target admission remain separate.')
CARD.write_bytes(card)
OUT.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k != 'requests'}, indent=2), flush=True)
