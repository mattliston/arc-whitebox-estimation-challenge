"""One native range-finder counterfactual, extracted from the bank's own history."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
protocol_path = HERE/'native_training_star_counterfactual_protocol_v1.json'
protocol = json.loads(protocol_path.read_text())
parent = HERE/protocol['parent']
assert hashlib.sha256(parent.read_bytes()).hexdigest() == protocol['parent_sha256']
text = parent.read_text()
lines = text.splitlines()
nodes = [n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == '_lr_factor']
assert len(nodes) == 2
node = nodes[0]
body = '\n'.join(lines[node.lineno-1:node.end_lineno])
assert '_LR_OMEGA[0]' in body and 'A32 @ om' in body and '_LRN_POWER' not in body
original_body = body
marker = '    n = int(A32.shape[0])'
assert body.count(marker) == 1
body = body.replace(marker, marker + "\n    assert n == 1024\n    _LRG_STATS['calls'] += 1")
assert body.replace("\n    assert n == 1024\n    _LRG_STATS['calls'] += 1", '') == original_body
fragment = "\n\n# Fixed training-range counterfactual; no other S31 semantics changed.\n_LRG_STATS = dict(calls=0)\n" + body + '\n'
candidate = HERE/'candidate_training_star_counterfactual_v1.py'
with candidate.open('x') as fp:
    fp.write(text+fragment)
record = dict(parent_sha256=protocol['parent_sha256'],candidate=candidate.name,
              candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              original_factor_lines=[node.lineno,node.end_lineno],
              original_factor_sha256=hashlib.sha256(original_body.encode()).hexdigest(),
              original_prefix_bytes_preserved=candidate.read_bytes()[:parent.stat().st_size] == parent.read_bytes(),
              changed_behavior='Random Gaussian initial range instead of norm-selected Nyström columns; same power iteration and SVD rank')
assert record['original_prefix_bytes_preserved']
with (HERE/'native_training_star_counterfactual_build_v1.json').open('x') as fp:
    json.dump(record,fp,indent=2)
print(json.dumps(record))
