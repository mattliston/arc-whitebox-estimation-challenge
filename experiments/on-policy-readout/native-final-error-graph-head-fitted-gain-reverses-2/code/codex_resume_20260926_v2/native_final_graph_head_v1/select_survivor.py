"""Apply the retention rule fixed before either fit started; no new label reads."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
addendum=json.loads((HERE/'banking_retention_addendum_v2.json').read_text())
assert not addendum['models_started'] and not addendum['check_labels_accessed']
assert addendum['protocol_sha256']==sha(HERE/'protocol.json')
assessment=json.loads((HERE/'assessment.json').read_text())
pipeline=json.loads((HERE/'pipeline_complete.json').read_text())
assert pipeline['all_groups_retired'] and pipeline['assessment_sha256']==sha(HERE/'assessment.json')
limits=addendum['paid_screen_gates'];gates={}
for mode in ('graph','control'):
    score=assessment[mode+'_vs_bank']
    gates[mode]=dict(raw=bool(score['raw_ratio']<=limits['raw_ratio_max']),
                    confidence=bool(score['ci95_ratio'][1]<limits['ci95_upper_ratio_max']),
                    wins=bool(score['wins']>=limits['wins_min']))
eligible=[mode for mode in gates if all(gates[mode].values())]
if assessment['passes_information_gate']:
    winner='graph'
elif len(eligible)==1:
    winner=eligible[0]
elif len(eligible)==2:
    winner='graph' if assessment['graph_vs_control']['raw_ratio']<=.99 else 'control'
else:
    winner=None
result=dict(survivor=winner,graph_information_gate=assessment['passes_information_gate'],
            paid_screen_gates=gates,eligible=eligible,
            assessment_sha256=sha(HERE/'assessment.json'),addendum_sha256=sha(HERE/'banking_retention_addendum_v2.json'),
            new_label_reads=False,valid40_accessed=False,submission_qualified=False,
            decision='Qualify the frozen '+winner+' head natively' if winner else
               'Close both fixed heads; no CHECK-guided epoch, sign, scale, feature or architecture rescue')
with (HERE/'survivor.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
