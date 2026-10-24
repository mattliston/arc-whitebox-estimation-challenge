"""Three exact source replacements with full inverse verification."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'lazy_rotation_objective_v1/candidate.py'
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = PARENT.read_text()
assert sha(PARENT)=='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
lines = source.splitlines(keepends=True)
nodes = ast.parse(source).body
edits = []
for name in ('_nonlinear_pair_batch','_nonlinear_terminal_live','_pair_series_fused'):
    selected = [n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(selected)==1
    node = selected[0]
    start = sum(map(len,lines[:node.lineno-1]))
    stop = sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    old = source[start:stop]
    if name=='_pair_series_fused':
        new = (HERE/'fused_helper.py').read_text().rstrip('\n')
    else:
        new = old
        for before,after in (
            ('    T31 = t4[:, None] * rho if k31 else None\n',''),
            ('    T31=T31.astype(fnp.float32)\n',''),
            ('_pair_series_fused(F,G,rho,T21,T22,T31,t3,t4,n)','_pair_series_fused(F,G,rho,T21,T22,None,t3,t4,n)')):
            assert new.count(before)==1
            new = new.replace(before,after,1)
    ast.parse(new)
    edits.append((start,stop,old,new,name))
candidate = source
for start,stop,old,new,name in sorted(edits,reverse=True):
    assert candidate[start:stop]==old
    candidate = candidate[:start]+new+candidate[stop:]
restored = candidate
for start,stop,old,new,name in sorted(edits):
    assert restored[start:start+len(new)]==new
    restored = restored[:start]+old+restored[start+len(new):]
assert restored==source
ast.parse(candidate)
with (HERE/'candidate.py').open('x') as f:f.write(candidate)
old_dir = HERE.parent.parent/'astra_phase2_reexamination_v1/current_pair_polynomial_fusion_v1'
record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
              parent_sha256=sha(PARENT),candidate_sha256=sha(HERE/'candidate.py'),
              fragment_sha256=sha(HERE/'fused_helper.py'),build_source_sha256=sha(__file__),
              protocol_sha256=sha(HERE/'PROTOCOL.md'),whole_source_inverse_byte_exact=True,
              replaced_functions=[e[-1] for e in edits],all_other_parent_source_unchanged=True,
              symmetric_rho_real_identity=True,asymmetry_preserved=True,all_layer_byte_equivalence_claimed=False,
              old_component_gate_remains_failed=True,
              old_reports={name:sha(old_dir/name) for name in
                 ('PROTOCOL.md','NATIVE_BLOCK_POSITIVE_20260913_UTC.md','RPC_COMPONENT_GATE_NEGATIVE_20260913_UTC.md')},
              full_model_executed=False,truth_accessed=False)
with (HERE/'build.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
