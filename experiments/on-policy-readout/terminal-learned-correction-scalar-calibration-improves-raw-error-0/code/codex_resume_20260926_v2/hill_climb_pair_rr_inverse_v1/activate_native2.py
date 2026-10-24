"""Bind the completed RR pilot; no numerical or model execution."""
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent
RR=D.parent/'hill_climb_pair_rr_identity_v1'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def load(p):return json.loads(Path(p).read_text())
def write(p,o):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(o,indent=2)+'\n');return pin(p)
def verify(o):
    if isinstance(o,dict):
        if all(k in o for k in ('path','sha256','bytes')):
            assert pin(o['path'])=={k:o[k] for k in ('path','sha256','bytes')},o['path']
        for v in o.values():verify(v)
    elif isinstance(o,list):
        for v in o:verify(v)
a=load(RR/'native2_assessment.json');rc=load(RR/'native_screen.json')
assert a['admitted'] and a['passed_integrity'] and all(a['gates'].values())
assert a['all16_parent_bytes_exact'] and a['all16_predictions_finite']
assert a['n']==2 and a['ranks']==[35,100] and a['parity_layers']==list(range(16))
assert a['config']==pin(RR/'native_screen.json') and a['candidate']==pin(RR/'candidate.py')
verify(a)
old_commands=load(RR/'commands_native2.json')
for name,key,limit in [('native2_guard.json','native2',150),('native2_assess_guard.json','native2_assessment',120)]:
    g=load(RR/name)
    assert g['command']==old_commands['commands'][key]
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None
    assert g['remaining_rows']==[] and g['elapsed_s']<=limit
    assert g['rss_trigger_bytes']==7500000000 and g['max_rss_bytes']<7500000000 and g['max_swap_growth_bytes']==0
cfg=load(D/'native_screen_draft.json');verify(cfg)
cfg['prospective_authorization']=True
cfg.pop('activation_required')
cfg['incremental_control_native2']=dict(source=pin(RR/'candidate.py'),assessment=pin(RR/'native2_assessment.json'),
    assessment_guard=pin(RR/'native2_assess_guard.json'),artifacts=a['blocks'][0]['arms']['candidate'],
    all16_files_byte_equal_required=True,mean_bill_strictly_lower_required=True,
    primary_comparator_unchanged=cfg['arms']['base']['source'])
cfg['interpretation']='Activated only after actual RR N2 PASS; primary exact gates compare official P128. Also require whole prediction-file byte equality and positive mean bill saving versus pinned actual RR N2. Own combined boundary/N8 required; no broad authority.'
cfg['provenance'] += [pin(D/'native_screen_draft.json'),pin(__file__),pin(RR/'native2_assessment.json'),pin(RR/'native2_guard.json'),pin(RR/'native2_assess_guard.json')]
c=write(D/'native_screen.json',cfg)
commands=load(D/'commands_native2_draft.json')
commands['config']=c
commands['commands']['native2_assessment'][3:5]=[c['path'],c['sha256']]
commands['guarded_commands']['native2_assessment'][-6:]=commands['commands']['native2_assessment']
commands['scope']='Root-only combined N2 now authorized after RR PASS; primary P128 exact gates plus pinned actual RR incremental check. No model executed by activation.'
cp=write(D/'commands_native2.json',commands)
print(json.dumps(dict(config=c,commands=cp,candidate=pin(D/'candidate.py')),indent=2))
