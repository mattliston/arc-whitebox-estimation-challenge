"""Version only trainability, retaining an exact inverse to the old model."""
import ast
from common import *

old=D/'current_full_lower_p_proposal_v1/model_v2.py'
original=old.read_text();text=original
edits=[
 ('requires_grad=trainable and k in PKEYS','requires_grad=False'),
 ('{k: torch.tensor(v) for k, v in e.items()}',
  '{k: torch.tensor(v, requires_grad=trainable and k in EKEYS) for k, v in e.items()}'),
 ('assert not a.requires_grad and not c.requires_grad and all(not x.requires_grad for x in edge.values())',
  "assert not a.requires_grad and not c.requires_grad and all(not x.requires_grad for x in p.values())\n        assert all(not x.requires_grad for k,x in edge.items() if k not in EKEYS)")]
for before,after in edits:
    assert text.count(before)==1,before
    text=text.replace(before,after)
inverse=text
for before,after in reversed(edits):
    assert inverse.count(after)==1
    inverse=inverse.replace(after,before)
assert inverse==original
ast.parse(text)
with (HERE/'model_edge.py').open('x') as f:f.write(text)
copies=[(D/'current_full_lower_p_allocator_v1/allocator.py','allocator.py'),
        (D/'current_p_dual_terminal_information_v1/guard.py','guard.py')]
for path,name in copies:
    with (HERE/name).open('xb') as f:f.write(path.read_bytes())
io=D/'current_p_batch_process_fit_v1/state_io.py'
iotext=io.read_text();assert iotext.count('from iso_common import *')==1
newio=iotext.replace('from iso_common import *','from common import *')
assert newio.replace('from common import *','from iso_common import *')==iotext
with (HERE/'state_io.py').open('x') as f:f.write(newio)
data=read(D/'current_full_lower_p_fit_v1/inputs.json')
weights={str(r):data['weights'][str(r)] for r in (35,100)}
for p in weights.values():check(p)
write(HERE/'inputs.json',dict(weights=weights,targets_decoded=False,source=pin(D/'current_full_lower_p_fit_v1/inputs.json')))
write(HERE/'build.json',dict(parent=pin(SOURCE),original_model=pin(old),model=pin(HERE/'model_edge.py'),
    whole_model_inverse_exact=True,only_trainability_checks_changed=True,edits=edits,
    state_io_inverse_exact=True,state_io_parent=pin(io),copied_sources=[pin(p) for p,n in copies],
    edge_trainable_values=528,parent_p_values_fixed=True,native_derivative_certified=False))
external=[SOURCE,SHARED,old,io,D/'current_full_lower_p_allocator_v1/allocator.py',
    D/'current_full_lower_p_allocator_v1/closeout.json',D/'current_p_batch_process_qualification_v1/qualification.json',
    D/'current_p_accuracy_history_audit_v1/RESULT.md',D/'current_p_coupling_history_audit_v1/RESULT.md',
    D/'current_p_native_gradient_history_audit_v1/RESULT.md',D/'covariance_head_spsa_native_v1/PROTOCOL.md',
    D/'covariance_head_spsa_native_v1/outcome.json',D/'current_p_dual_terminal_information_v1/closeout.json']
paths=external+[p for p in HERE.iterdir() if p.is_file()]+[Path(p['path']) for p in weights.values()]
pins=[pin(p) for p in sorted(set(paths))]
write(HERE/'sources_sealed.json',dict(pins=pins,targets_decoded=False,protocol=pin(HERE/'PROTOCOL.md')))
print(dict(prepared=True,pins=len(pins),only_528_edge_values_train=True),flush=True)
