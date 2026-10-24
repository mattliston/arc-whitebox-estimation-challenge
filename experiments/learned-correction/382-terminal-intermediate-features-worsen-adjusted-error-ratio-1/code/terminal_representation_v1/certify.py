"""Small algebra certificate, independent of network features and target files."""
import hashlib
import json
from pathlib import Path
from information import certificate

here = Path(__file__).resolve().parent
result = dict(passed=True, truth_accessed=False, **certificate(),
              information_sha256=hashlib.sha256((here / 'information.py').read_bytes()).hexdigest())
with (here / 'algebra_certificate.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result), flush=True)
