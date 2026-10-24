"""Isolate summation-order research from coefficient reassociation; never ship it."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
base = HERE / 'candidate_bundle_diagreuse_v1.py'
assert hashlib.sha256(base.read_bytes()).hexdigest() == '5699a9a241722cc1f8ef7f94a5cbffb82c2b00e4cfdc29e41f330ebed1f57cfc'
fragment = (HERE / 'pair_order_fragment_v1.py').read_text()
old = '''    weighted = left_squared[:, perm] * coefficients[perm][None, :]
    return _shared_old384_mm(weighted, factors.T)'''
new = '''    weighted_factors = factors * coefficients[perm][None, :]
    return _shared_old384_mm(left_squared[:, perm], weighted_factors.T)'''
assert fragment.count(old) == 1
fragment = fragment.replace(old, new)
with (HERE / 'pair_order_fragment_v2.py').open('x') as f:
    f.write(fragment)
candidate = base.read_text() + '\n\n' + fragment
with (HERE / 'candidate_pair_order_research_v2.py').open('x') as f:
    f.write(candidate)
test = (HERE / 'verify_pair_order.py').read_text()
test = test.replace('candidate_pair_order_research_v1.py', 'candidate_pair_order_research_v2.py')
test = test.replace('pair_order_algebra_v1.json', 'pair_order_algebra_v2.json')
with (HERE / 'verify_pair_order_v2.py').open('x') as f:
    f.write(test)
record = dict(
    base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
    fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    status='prepared, no native test; excluded from S30 qualification',
    difference_from_v1='Coefficients stay on right factors, matching the baseline reader. Only column ordering changes.')
with (HERE / 'pair_order_research_manifest_v2.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
