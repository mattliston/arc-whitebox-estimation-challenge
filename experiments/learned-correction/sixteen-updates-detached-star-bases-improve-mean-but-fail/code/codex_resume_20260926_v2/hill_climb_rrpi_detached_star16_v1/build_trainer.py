"""Prepare source only: resume the unchanged detached-star trainer from8 to16."""
from pathlib import Path
import ast, difflib, hashlib, json
D = Path(__file__).resolve().parent
OLD = D.parent / 'hill_climb_rrpi_detached_star_v1'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,s):
    with p.open('x') as f:f.write(s)
s=(OLD/'train_segment.py').read_text()
edits=[
 ('"""Fixed RRPI detached-star eight-update trial in two optimizer-preserving segments."""',
  '"""Fixed detached-star continuation: resume original update8 through update16."""'),
 ('assert 0 <= SEGMENT < 2','assert 2 <= SEGMENT < 4'),
 ("assert protocol['updates'] == 8 and protocol['batch'] == 4 and protocol['schedule_steps'] == 120",
  "assert protocol['updates'] == 16 and protocol['batch'] == 4 and protocol['schedule_steps'] == 120\nassert protocol['continuation']['segment_start'] == 2 and protocol['continuation']['segment_stop'] == 4"),
 ("    previous = HERE / ('resume_after_%03d.pt' % START)\n    prior = json.loads((HERE / ('segment_%02d' % (SEGMENT-1)) / 'completed.json').read_text())",
  "    previous_root = Path(protocol['continuation']['prior_protocol']['path']).parent if START == 8 else HERE\n    previous = previous_root / ('resume_after_%03d.pt' % START)\n    prior = json.loads((previous_root / ('segment_%02d' % (SEGMENT-1)) / 'completed.json').read_text())\n    expected_protocol_sha = protocol['continuation']['prior_protocol']['sha256'] if START == 8 else sys.argv[1]\n    if START == 8:\n        assert pin(previous) == protocol['continuation']['prior_checkpoint']\n        assert pin(previous_root / 'segment_01/completed.json') == protocol['continuation']['prior_segment_completion']\n    assert prior['protocol']['sha256'] == expected_protocol_sha"),
 ("    assert sv['update'] == START and sv['protocol_sha'] == sys.argv[1]",
  "    assert sv['update'] == START and sv['protocol_sha'] == expected_protocol_sha"),
 ('trainer=pin(__file__), ranks=RANKS, updates=8, batch=4, schedule_steps=120, segment=SEGMENT, start=START, stop=STOP,',
  'trainer=pin(__file__), ranks=RANKS, updates=16, batch=4, schedule_steps=120, segment=SEGMENT, start=START, stop=STOP,'),
 ('if STOP == 8:', 'if STOP == 16:'),
 ('endpoint_selected=(STOP == 8), native_gradient_certified=False))',
  'endpoint_selected=(STOP == 16), native_gradient_certified=False))'),
 ('final_endpoint=STOP==8)), flush=True)', 'final_endpoint=STOP==16)), flush=True)'),
]
t=s
for a,b in edits:
    assert t.count(a)==1,a
    t=t.replace(a,b,1)
back=t
for a,b in reversed(edits):
    assert back.count(b)==1,b
    back=back.replace(b,a,1)
assert back==s
compile(ast.parse(t),str(D/'train_segment.py'),'exec')
write(D/'train_segment.py',t)
copies=[]
for n in ('native_parameters.py','data_roster.json'):
    with (D/n).open('xb') as f:f.write((OLD/n).read_bytes())
    assert (D/n).read_bytes()==(OLD/n).read_bytes()
    copies.append(dict(parent=pin(OLD/n),copy=pin(D/n),byte_identical=True))
write(D/'trainer.diff',''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile=str(OLD/'train_segment.py'),tofile=str(D/'train_segment.py'))))
record=dict(passed=True,source_only=True,builder=pin(__file__),trainer=pin(D/'train_segment.py'),diff=pin(D/'trainer.diff'),
 source_inverse=dict(parent=pin(OLD/'train_segment.py'),edits=[dict(before=a,after=b) for a,b in edits],exact_inverse=True),copies=copies,
 scope='Only endpoint/segment/resume-provenance changes. Forward and detached-gradient arithmetic, optimizer, dials, first120 cosine and original320-example order remain unchanged.',
 prior_checkpoint=pin(OLD/'resume_after_008.pt'),prior_completion=pin(OLD/'training_complete.json'),
 original_failure=pin(OLD/'native_assessment.json'),adaptive_followup=pin(OLD/'followup_assessment.json'),
 model_or_numerical_execution=False)
write(D/'trainer_build.json',json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(trainer=record['trainer'],build=pin(D/'trainer_build.json'))))
