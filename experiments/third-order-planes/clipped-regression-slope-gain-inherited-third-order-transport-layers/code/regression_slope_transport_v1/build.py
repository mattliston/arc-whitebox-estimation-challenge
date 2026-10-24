"""Append isolated active functions; preserve complete original parent bytes."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=D/'lazy_rotation_objective_v1/candidate.py'
source=parent.read_text()
assert sha(parent)=='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
lines=source.splitlines()
functions={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
def function(name):
    node=functions[name]
    return '\n'.join(lines[node.lineno-1:node.end_lineno])

pair=function('_nonlinear_pair_batch')
pair_new=pair.replace('def _nonlinear_pair_batch(', 'def _rgt_pair_with_raw2(',1)
anchor='post_v=var, post_k3=post_k3, post_k4=post_k4)'
replacement='post_v=var, post_k3=post_k3, post_k4=post_k4, raw2=r2)'
assert pair_new.count(anchor)==1
pair_new=pair_new.replace(anchor,replacement,1)
assert pair_new.replace(replacement,anchor,1).replace('def _rgt_pair_with_raw2(','def _nonlinear_pair_batch(',1)==pair

old=function('_predict_binary')
edits=[('            cp_post = (inherited + born) * live',
        '            gcp, inherited_cp = _rgt_rebase_history(l, mu, host, aux, cp_pre, inherited)\n            cp_post = (inherited_cp + born) * live\n            del inherited_cp'),
       ('            p = g.astype(fnp.float32)[:, None] * pp',
        '            p = gcp.astype(fnp.float32)[:, None] * pp'),
       ('            _shared_old384_commit(l, g, alive, selected, shared_old)',
        '            _shared_old384_commit(l, gcp, alive, selected, shared_old)')]
new=old
for a,b in edits:
    assert new.count(a)==1
    new=new.replace(a,b,1)
inverse=new
for a,b in reversed(edits):
    assert inverse.count(b)==1
    inverse=inverse.replace(b,a,1)
assert inverse==old
cut=old.index(edits[0][0]);newcut=new.index('            gcp, inherited_cp =')
assert old[:cut]==new[:newcut]
assert old.index('innovation =')<old.index('born = born + _bd_plane_delta')<cut
kernel=(HERE/'kernel.py').read_text()
extension='\n\n# Fixed empirical regression-slope history transport.\n'+kernel+'\n\n'+pair_new+'\n\nnonlinear = _rgt_pair_with_raw2\n\n'+new+'\n'
candidate=source+extension
assert candidate[:-len(extension)]==source
ast.parse(candidate)
for name,body in (('candidate.py',candidate),('null.py',candidate+'\n_RGT_ENABLED = False\n')):
    with (HERE/name).open('x') as stream:stream.write(body)
record=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            parent_sha256=sha(parent),candidate_sha256=sha(HERE/'candidate.py'),
            null_sha256=sha(HERE/'null.py'),kernel_sha256=sha(HERE/'kernel.py'),
            parent_byte_prefix_exact=True,source_inverse_byte_exact=True,
            aux_raw2_only_inverse_exact=True,source_through_newborn_exact=True,
            derivative_correctness_claim=False,mixed_coefficient_counterexample_retained=True,
            model_executed=False,truth_accessed=False)
with (HERE/'build.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
