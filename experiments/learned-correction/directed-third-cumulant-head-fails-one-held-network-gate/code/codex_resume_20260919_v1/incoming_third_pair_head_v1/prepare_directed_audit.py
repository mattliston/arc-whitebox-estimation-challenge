import ast
from datetime import datetime,timezone
from common import *
verify();verify('fit_sources_sealed.json')
names=['DIRECTED_RESPONSE_CORRECTION.md','directed_response_audit.py','prepare_directed_audit.py']
for n in names:
    if n.endswith('.py'):ast.parse((HERE/n).read_text())
write(HERE/'directed_audit_sealed.json',dict(utc=datetime.now(timezone.utc).isoformat(),pins=[pin(HERE/n) for n in names]+[
      pin(A/'PAIR_MEMORY_RESPONSE_EXACTNESS_V2.md')],new_fit=False,model=pin(HERE/'model.npz'),
      heldout_information_result_read_by_root_before_seal=False,original_primary_gates_unchanged=True))
