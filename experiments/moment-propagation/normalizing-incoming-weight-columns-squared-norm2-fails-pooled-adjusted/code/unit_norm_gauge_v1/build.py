"""Add one same-network wrapper and prove all other parent source unchanged."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'lazy_rotation_objective_v1/candidate.py'
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw = PARENT.read_bytes()
assert sha(PARENT)=='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
node = [n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Estimator'][-1]
lines = raw.splitlines(keepends=True)
start = sum(map(len,lines[:node.lineno-1]))+node.col_offset
stop = sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
old = raw[start:stop]
before,after = b'return _predict_learned_compact(mlp)',b'return _unitnorm_predict(mlp)'
assert old.count(before)==1
changed = old.replace(before,after)
assert changed.replace(after,before)==old
fragment = (HERE/'gauge_fnp.py').read_bytes()
assert all(isinstance(n,(ast.Expr,ast.FunctionDef,ast.ClassDef)) for n in ast.parse(fragment).body)
delimiter = b'\n\n# BEGIN FIXED INCOMING-COLUMN GAUGE\n'
candidate = raw[:start]+changed+raw[stop:]+delimiter+fragment
recovered = candidate[:-len(delimiter)-len(fragment)]
recovered = recovered[:start]+old+recovered[start+len(changed):]
assert recovered==raw
ast.parse(candidate)
with (HERE/'candidate.py').open('xb') as f:f.write(candidate)
rec = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
           parent_sha256=sha(PARENT),candidate_sha256=sha(HERE/'candidate.py'),
           fragment_sha256=sha(HERE/'gauge_fnp.py'),build_source_sha256=sha(__file__),
           protocol_sha256=sha(HERE/'PROTOCOL.md'),whole_source_inverse_byte_exact=True,
           final_estimator_call_edits=1,all_other_parent_source_bytes_unchanged=True,
           hidden_column_squared_norm=2,final_scale=1,scale_dtype='float64',
           transformed_weight_dtype='float32',full_model_executed=False,truth_accessed=False)
with (HERE/'build.json').open('x') as f:json.dump(rec,f,indent=2)
print(json.dumps(rec))
