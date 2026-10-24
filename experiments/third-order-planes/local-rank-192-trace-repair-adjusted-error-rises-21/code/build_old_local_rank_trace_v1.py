"""Build a rank-192 local approximation on frozen S31, training screen only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent = HERE/'candidate_old_both_tail_trace_bounded_v6.py'
sha = '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
source = parent.read_text()
assert hashlib.sha256(parent.read_bytes()).hexdigest() == sha
assert json.loads((HERE/'old_local_rank_geometry_v1.json').read_text())['passed']
node = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='omitted_radial_source'][-1]
radial = ast.get_source_segment(source,node)
start = radial.index('    if _OTT_DATA:')
end = radial.index('    contraction =',start)
radial = radial[:start] + '''    if _OTT_DATA:
        delta_energy, delta_sum = _lrr_radial_delta(a, beta)
        columns = int(delta_energy.shape[0])
        _ott_h0 = fnp.add(row_energy[:columns], delta_energy, out=row_energy[:columns])
        _ott_h1 = fnp.add(column_sum[:columns], delta_sum, out=column_sum[:columns])
''' + radial[end:]
fragment = HERE/'old_local_rank_trace_fragment_v1.py'
source += '\n\n' + fragment.read_text() + '\n\n' + radial + '\n'
target = HERE/'candidate_old_local_rank192_trace_v1.py'
compile(source,str(target),'exec')
with target.open('x') as f:f.write(source)
record = dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate=str(target),
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(),parent=str(parent),parent_sha256=sha,
              fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),local_read_rank=192,
              changed='Shorten only the kept local old-row reconstruction at L7-14; trace(full)-trace(short) diagonal and radial corrections.',
              preserved='S31 tail roster and tail trace; full propagated factors; all young columns; setup keys and trained parameters.',
              approximation='The squared left operand of the old response-plane contraction is approximate.',
              cost='Lazy canonical factors explicitly materialized, metered and cached only until the next join.',
              protocol=dict(initial_ranks=[35,100],baseline='both_tail_train2_v6',
                            expand_only_if='TRAIN2 adjusted gain at least 0.5%; then matched TRAIN6 before any validation'),
              status='Unqualified training-only research')
with (HERE/'old_local_rank_manifest_v1.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
