"""Apply the retention rule fixed before either fit started; no new label reads."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
addendum=json.loads((HERE/'control_retention_addendum_v1.json').read_text())
assert not addendum['models_started'] and not addendum['check_labels_accessed']
assert addendum['protocol_sha256']==sha(HERE/'protocol.json')
assessment=json.loads((HERE/'assessment.json').read_text())
pipeline=json.loads((HERE/'pipeline_complete.json').read_text())
assert pipeline['all_groups_retired'] and pipeline['assessment_sha256']==sha(HERE/'assessment.json')
score=assessment['control_vs_bank'];limits=addendum['control_gates']
control_gates=dict(raw=bool(score['raw_ratio']<=limits['raw_ratio_max']),
                   confidence=bool(score['ci95_ratio'][1]<limits['ci95_upper_ratio_max']),
                   wins=bool(score['wins']>=limits['wins_min']))
winner='graph' if assessment['passes_information_gate'] else 'control' if all(control_gates.values()) else None
result=dict(survivor=winner,graph_information_gate=assessment['passes_information_gate'],
            control_gates=control_gates,control_eligible=all(control_gates.values()),
            assessment_sha256=sha(HERE/'assessment.json'),addendum_sha256=sha(HERE/'control_retention_addendum_v1.json'),
            new_label_reads=False,valid40_accessed=False,submission_qualified=False,
            decision='Qualify the frozen '+winner+' head natively' if winner else
               'Close both fixed heads; no CHECK-guided epoch, sign, scale, feature or architecture rescue')
with (HERE/'survivor.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
