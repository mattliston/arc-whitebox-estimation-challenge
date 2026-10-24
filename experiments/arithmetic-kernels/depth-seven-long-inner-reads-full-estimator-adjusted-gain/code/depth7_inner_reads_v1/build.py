"""Build the unchanged arithmetic as an explicit full-program diagnostic."""
from pathlib import Path
import ast
import hashlib
import json

D=Path(__file__).resolve().parent
H=D.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((D/'source_seal.json').read_text())
for name,value in seal['files'].items():
    assert sha(D/name)==value,name
g=json.loads((D/'bench_guard.json').read_text())
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
r=json.loads((D/'bench_result.json').read_text())
assert r['gates']['relative_error'] and not r['passes_primitive_gate']
parent=H/'candidate_old_both_tail_trace_bounded_v6.py'
assert sha(parent)==seal['parent_sha256']
lifecycle='''
# Participate in the bank's scratch-reset lifecycle; no numerical change.
_D7R_A, _D7R_B, _D7R_C = _D7R_CACHES
_PB_CACHES = _PB_CACHES + ('_D7R_A', '_D7R_B', '_D7R_C')
'''
source=parent.read_text()+'\n\n'+(D/'fragment.py').read_text()+lifecycle
ast.parse(source)
with (D/'candidate.py').open('x') as f:f.write(source)
# The evaluator only adds routing telemetry to the established native scorer.
old=(H/'native_eval.py').read_text()
old=old.replace('HERE=Path(__file__).resolve().parent','HERE=Path(__file__).resolve().parent.parent',1)
needle="p6=dict(getattr(mod,'_P6_STATS',{})))"
assert old.count(needle)==1
new=old.replace(needle,"p6=dict(getattr(mod,'_P6_STATS',{})),d7r=dict(getattr(mod,'_D7R_STATS',{})))")
ast.parse(new)
with (D/'evaluate.py').open('x') as f:f.write(new)
files=['candidate.py','fragment.py','build.py','evaluate.py','FULL_PROGRAM_FOLLOWUP.md','source_seal.json']
with (D/'pre_endpoint_seal.json').open('x') as f:
    json.dump(dict(files={name:sha(D/name) for name in files},parent_sha256=sha(parent),
        primitive_gate_passed=False,explicit_full_program_followup=True,
        ranks=[35,100],adjusted_gain_min=.015,raw_degradation_max=.005,
        targets_accessed=False),f,indent=2)
print(json.dumps(dict(candidate_sha256=sha(D/'candidate.py'),bytes=len(source.encode()),
    pre_endpoint_sha256=sha(D/'pre_endpoint_seal.json'))))
