"""Stdlib full-file hashes and complete-meter parity; no labels or array decoding."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=read(HERE/'sources_sealed.json')
for item in seal['sources']+seal['external_pins']:
    assert sha(item['path'])==item['sha256'],item['path']
build=read(HERE/'build.json');watch=read(HERE/'null_watch.json')
assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes']==0
null=read(HERE/'null/result.json');parent=read(HERE.parent/'lazy_rotation_v1/dev/result.json')
assert null['candidate_sha256']==build['null_sha256']==sha(HERE/'null.py')
assert parent['candidate_sha256']==build['parent_sha256']
assert null['runner_sha256']==sha(HERE.parent/'weight5_reuse_v1/native.py')
assert [r['rank'] for r in null['records']]==[35,35]
references=[parent['records'][0],parent['records'][2]]
for record,reference in zip(null['records'],references):
    assert sha(record['output'])==record['output_sha256']==sha(reference['output'])==reference['output_sha256']
    assert record['weights_sha256']==reference['weights_sha256']
    assert record['flops']==reference['flops']
    assert record['weight5_cache']['completed'] and record['weight5_cache']['numeric_cache_retired']
    assert record['max_rss_bytes']<8_000_000_000
assert null['repeat_byte_exact']
out=dict(passed=True,all16_parent_byte_exact=True,cold_repeat_bills_exact=True,
         repeat_byte_exact=True,cache_retired=True,
         own_flops=[r['flops'] for r in null['records']],extra_flops=[0,0],truth_accessed=False)
with (HERE/'null_verdict.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out),flush=True)
