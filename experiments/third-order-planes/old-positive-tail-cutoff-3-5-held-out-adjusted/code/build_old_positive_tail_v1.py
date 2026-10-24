from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
base = HERE / 'candidate_bundle_diagreuse_v1.py'
sha = hashlib.sha256(base.read_bytes()).hexdigest()
assert sha == '5699a9a241722cc1f8ef7f94a5cbffb82c2b00e4cfdc29e41f330ebed1f57cfc'
fragment = HERE / 'old_positive_tail_fragment_v1.py'
candidate = HERE / 'candidate_old_positive_tail_v1.py'
text = base.read_text() + '\n\n' + fragment.read_text()
compile(text, str(candidate), 'exec')
with candidate.open('x') as f:
    f.write(text)
record = dict(base_sha256=sha, fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),
              candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
              threshold=3.5, layers=list(range(7, 15)),
              status='research approximation; native accuracy and cost untested')
with (HERE / 'old_positive_tail_manifest_v1.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
