"""Versioned composition: unchanged Cost150 plus one fixed older-only tier."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent = HERE / 'cost_rule150_scalar.py'
fragment = HERE / 'nested224_fragment.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent) == 'f06314111d7f41a7c1fbcb8d888a509d0e46222cd31b5ab227ec1e334e582a1f'
source = parent.read_text()
added = '\n\n' + fragment.read_text()
candidate = source + added
ast.parse(candidate)
assert candidate[:-len(added)] == source
out = HERE / 'nested224_cost150.py'
with out.open('x') as f:
    f.write(candidate)
record = dict(parent_sha256=sha(parent), fragment_sha256=sha(fragment),
              candidate_sha256=sha(out), source_inverse_exact=True,
              first_tier_rank=384, second_tier_rank=224, moves=list(range(8, 15)),
              learned_payloads_changed=False, labels_used=False,
              note='Approximate tensor compression; actual accuracy/cost unknown.')
with (HERE / 'nested224_build.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
