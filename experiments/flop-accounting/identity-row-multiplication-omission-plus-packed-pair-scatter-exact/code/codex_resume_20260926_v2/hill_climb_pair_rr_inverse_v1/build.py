"""Source-only composition of RR identity and the unchanged retained PI delta."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

D=Path(__file__).resolve().parent
H=D.parent
RR=H/'hill_climb_pair_rr_identity_v1'
PI=H/'hill_climb_packed_pair_inverse_v1'

def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def load(p): return json.loads(Path(p).read_text())
def write(name,obj):
    p=D/name; assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,indent=2)+'\n'); return pin(p)
def apply(text,edits):
    for e in edits:
        assert text.count(e['before'])==1
        text=text.replace(e['before'],e['after'])
    return text
def undo(text,edits):
    for e in reversed(edits):
        assert text.count(e['after'])==1
        text=text.replace(e['after'],e['before'])
    return text

pi=load(PI/'source_inverse.json'); rr=load(RR/'build.json')
assert pin(PI/'candidate.py')['sha256']=='655112635569ff371add902b710201103005aaec2482d62a323f86059a472e66'
assert pin(RR/'candidate.py')['sha256']=='7b53cf6a14032cdf84542cd3d56442fcf543ceba6044a2819df0da24d112d91a'
assert pi['suffix']=='\n'+(PI/'fragment.py').read_text()
assert apply(Path(pi['parent']['path']).read_text(),pi['edits'])+pi['suffix']==(PI/'candidate.py').read_text()
parent=(RR/'candidate.py').read_text()
candidate=apply(parent,pi['edits'])+pi['suffix']
assert undo(candidate[:-len(pi['suffix'])],pi['edits'])==parent
assert undo(parent,rr['source_inverse']['edits'])==Path(rr['source_inverse']['parent']['path']).read_text()
compile(candidate,str(D/'candidate.py'),'exec')
prior_ast=ast.parse(parent); new_ast=ast.parse(candidate[:-len(pi['suffix'])])
changed=[getattr(a,'name',type(a).__name__) for a,b in zip(prior_ast.body,new_ast.body) if ast.dump(a)!=ast.dump(b)]
assert len(prior_ast.body)==len(new_ast.body) and changed==['_psp_pair_series']
assert not (D/'candidate.py').exists()
(D/'candidate.py').write_text(candidate)
(D/'source.diff').write_text(''.join(difflib.unified_diff(parent.splitlines(True),candidate.splitlines(True),fromfile=str(RR/'candidate.py'),tofile=str(D/'candidate.py'))))
inverse=write('source_inverse.json',dict(parent=pin(RR/'candidate.py'),edits=pi['edits'],suffix=pi['suffix'],
    fragment=pin(PI/'fragment.py'),exact_inverse=True,rr_to_p128=pin(RR/'build.json'),
    retained_pi_inverse=pin(PI/'source_inverse.json'),retained_pi_candidate=pin(PI/'candidate.py')))
cfg=load(RR/'native_screen.json')
protocol=write('protocol.json',dict(schema='conditional-RR-plus-retained-PI-v1',candidate=pin(D/'candidate.py'),
    source_parent=pin(RR/'candidate.py'),score_comparator=cfg['arms']['base']['source'],
    source_inverse=inverse,rr_protocol=pin(RR/'protocol.json'),pi_retained=pin(PI/'retained_component.json'),
    pi_native8=pin(PI/'native8_assessment.json'),pi_primitive=pin(PI/'primitive.json'),
    pi_prior_incremental_adjusted_pct=-0.04529975341369674,pi_prior_all16_exact=True,
    scope='Exact retained PI edit and unchanged fragment transplanted onto RR/P128. No other source edits; no setup, routing, rank, table or cache-budget change.',
    mechanism='RR omits multiplication by exact unity; PI replaces five ordered packed takes and three P16-channel takes using one paid int32 inverse, shared transpose view and discarded sentinel. Both preserve scalar/reduction order.',
    caveats=['Prior PI retained evidence is versus FG, not a claim that the current composition passes.',
             'One 4MiB inverse per prediction plus paid cold construction, P16 scratch and backend intp/flatten temporaries. Prior source estimate about6.95MB owned extra memory is not an RSS bound.',
             'PI public scatter and metadata checks alter dispatch count; own combined normal boundary/RSS/residual/wall and N8 are mandatory. Official FG failure remains preserved; current P128 is official S35.'],
    native2_gates=cfg['stages']['2']['gates'],native8_gates=cfg['stages']['8']['gates'],parity_layers=list(range(16)),
    activation_required='RR native2 assessment PASS plus clean model/assessment guards before this draft can be activated.',
    auxiliary_native2_required=dict(comparator_source=pin(RR/'candidate.py'),
        assessment_path=str(RR/'native2_assessment.json'),tag=RR.name+'_train2_v1',ranks=[35,100],
        all16_prediction_files_byte_equal=True,mean_bill_strictly_lower=True,
        interpretation='Positive incremental cost saving and identical outputs versus actual RR N2; primary score gates remain against P128.'),
    authority='Conditional N2-only source preparation; no model execution. Combined own N2/boundary/N8 replaces separate RR N8. No broad/release authority; final >=1% over official S35 remains.',
    model_executed=False))
build=write('build.json',dict(candidate=pin(D/'candidate.py'),source_parent=pin(RR/'candidate.py'),
    native_comparator=cfg['arms']['base']['source'],source_inverse=inverse,
    protocol=protocol,builder=pin(__file__),source_diff=pin(D/'source.diff'),
    only_changed_parent_functions=changed,exact_rr_and_pi_inverse=True,ast_compile=True))
cfg['arms']['candidate']['source']=pin(D/'candidate.py')
for n in ('2','6'):
    b=cfg['arms']['candidate']['blocks'][n]
    b['tag']=D.name+'_train'+n+'_v1';b['guard']=str(D/('native'+n+'_guard.json'));b.pop('artifacts',None)
for n in ('2','8'):
    s=cfg['stages'][n];s['result']=str(D/('native'+n+'_assessment.json'));s['assessment_guard']=str(D/('native'+n+'_assess_guard.json'))
cfg['provenance']=[build,protocol,inverse,pin(RR/'native_screen.json'),pin(PI/'retained_component.json'),pin(PI/'primitive.json'),pin(H/'official_bank_S35_v1.json')]
cfg['boundary_before8']['receipt']=str(D/'NATIVE8_NOT_AUTHORIZED.json')
cfg['boundary_before8']['assessment_guard']=str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
cfg['prospective_authorization']=False
cfg['activation_required']=dict(rr_native2_assessment=str(RR/'native2_assessment.json'),must_pass=True)
cfg['interpretation']='INACTIVE conditional combined exact pilot; activate only after actual RR N2 PASS. Source comparison is P128; actual RR N2 incremental comparison also required.'
draft=write('native_screen_draft.json',cfg)
old=load(RR/'commands_native2.json')
commands=copy.deepcopy(old)
commands['config']=draft
commands['commands']['native2'][3]=str(D/'candidate.py')
commands['commands']['native2'][5]=D.name+'_train2_v1'
commands['commands']['native2_assessment'][3:5]=[draft['path'],draft['sha256']]
for key,seconds,record in [('native2',150,D/'native2_guard.json'),('native2_assessment',120,D/'native2_assess_guard.json')]:
    commands['guarded_commands'][key]=old['guarded_commands'][key][:3]+['--record',str(record),'--seconds',str(seconds),'--rss-limit','7500000000']+commands['commands'][key]
commands.update(stdout_log=str(D/'native2.log'),native2_guard=str(D/'native2_guard.json'),assessment_guard=str(D/'native2_assess_guard.json'),scope='INACTIVE: draft assessment deliberately rejects; do not launch native2 until root confirms RR PASS and activates.',executed=False)
commandpin=write('commands_native2_draft.json',commands)
def verify(o):
    if isinstance(o,dict):
        if all(k in o for k in ('path','sha256','bytes')):
            assert pin(o['path'])=={k:o[k] for k in ('path','sha256','bytes')},o['path']
        for v in o.values():verify(v)
    elif isinstance(o,list):
        for v in o:verify(v)
verify(cfg)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),draft=draft,commands=commandpin,build=build),indent=2))
