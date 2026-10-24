"""Version one fixed native CP-entry metric change without editing the bank."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;H=HERE.parent
parent=H/'candidate_old_both_tail_trace_bounded_v6.py'
src=parent.read_text();assert hashlib.sha256(parent.read_bytes()).hexdigest()=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
lines=src.splitlines(keepends=True)
nodes=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'
       and '_need_core' in ''.join(lines[n.lineno-1:n.end_lineno])]
assert len(nodes)==1;n=nodes[0]
body=''.join(lines[n.lineno-1:n.end_lineno])
old='''        coefficient = lam[start:stop]
        weight = coefficient * coefficient'''
new='''        coefficient = lam[start:stop]
        # Fixed scale-invariant diagonal-atom CP entry metric; paid reductions.
        _cm_j2 = _e2_H(J * J)
        _cm_norm2 = _e2_H(fnp.sum(_cm_j2, axis=0))
        weight = _e2_H(coefficient * coefficient) * _e2_H(_cm_norm2 * _cm_norm2)'''
assert body.count(old)==1;body=body.replace(old,new)
old='_jn_d = fnp.matmul(_e2_H(J * J), weight)'
assert body.count(old)==1;body=body.replace(old,'_jn_d = fnp.matmul(_cm_j2, weight)')
old='del J, weight, coefficient, Fj, joined_core, factors, core'
assert body.count(old)==1;body=body.replace(old,old+', _cm_j2, _cm_norm2')
newsrc=''.join(lines[:n.lineno-1])+body+''.join(lines[n.end_lineno:]);ast.parse(newsrc)
candidate=HERE/'candidate.py'
with candidate.open('x') as f:f.write(newsrc)
files=[HERE/x for x in ['PROTOCOL.md','build.py','unit.py','assess.py','candidate.py']]
files += [parent,H/'native_eval.py',H/'cp_scale_invariant_metric_protocol_v1.md',H/'cp_scale_invariant_metric_information_v1.json',
          H/'analyze_cp_metric_v1.py',H/'both_tail_train2_v6/rows.jsonl',H/'both_tail_train6_v6_routing_control/rows.jsonl']
seal=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),new_candidate_evaluated=False,
          candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
with (HERE/'seal.json').open('x') as f:json.dump(seal,f,indent=2);f.write('\n')
print(json.dumps(dict(candidate_sha256=seal['candidate_sha256'],single_function_changed=True,existing_information_gate_passed=False)))
