"""Preserve the exact scope of current frozen coupling and historical joint fits."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
JOINT=A/'current_comp_joint_p_node_v1'
FIT=D/'current_full_lower_p_fit_v1'
MODEL=D/'current_full_lower_p_proposal_v1/model_v2.py'
BANK=D/'full_lower_covariance_edge_packed_v1/candidate.py'


def read(path):return json.loads(path.read_text())
def pin(path):
    path=path.resolve();b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


assert pin(BANK)['sha256']=='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
assert pin(MODEL)['sha256']=='278112f097d0748c255869104a22297b248ba1c424c0c4528b6b6df04942c00e'
assert pin(FIT/'fit_result.json')['sha256']=='7c06cb7d16d98e53d1a9672df82571160bf525e44d254cbf02e939f638891797'
assert not (D/'current_p_native_retirement_v1/native40_score.json').exists()
source=BANK.read_text();tree=ast.parse(source)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_bdnh_coefficients')
excerpt=ast.get_source_segment(source,function)
assert 'H.shape != (n, 64) or A.shape != (64, 4)' in excerpt
assert 'response = fnp.tanh(standardized @ A)' in excerpt
with (HERE/'current_birth_node_function.py').open('x') as f:f.write(excerpt+'\n')
model=MODEL.read_text()
assert 'requires_grad=trainable and k in PKEYS' in model
assert 'assert not a.requires_grad and not c.requires_grad and all(not x.requires_grad for x in edge.values())' in model
assert "f['_bdnh_coefficients'](hs, self.A, self.cstar[layer])" in model
fit=read(FIT/'fit_result.json')
assert fit['completed'] and fit['training_graphs']==160 and fit['updates']==40
assert fit['fixed_parameters_unchanged'] and fit['inactive_parameters_unchanged']
score_path=JOINT/'score_v1/result.json'
assert pin(score_path)['sha256']=='38060450e770c2c66310d05919086b6fa3177d2461077587b2db61068b1e3b4b'
score=read(score_path)
assert score['completed'] and score['integrity_passed'] and score['exact_COMP_score_replay']
assert not score['joint_learning_lead'] and score['p_only_component']
assert not score['official_score'] and not score['independent_panel_qualified'] and not score['production_qualified']
training=[]
for role,sha in [('p_only','c2ce62dc3565e6d77ac14e8e6110348bd364eeec59cb0e9378500bd1348c3157'),
                 ('joint','1293ab4af3f75dc21fcb02000c7933a68aab0efdacd333d60ee6bb0b6e95fe4c')]:
    path=JOINT/f'train_{role}_v1/result.json'
    assert pin(path)['sha256']==sha
    r=read(path)
    assert r['completed'] and r['passed'] and r['epochs']==3 and r['updates']==15
    assert r['full_model_calls']==r['retired_call_count']==64 and r['training_graphs']==60
    assert not r['validation_targets_accessed'] and not r['native_gradient_qualified']
    training.append(dict(arm=role,result=pin(path),epochs=3,updates=15,training_graphs=60,
        train_ranks=r['train_ranks'],validation_ranks=r['validation_ranks'],trainable_keys=r['trainable_keys'],
        initial_A=r['initial_A'],terminal_A=r['terminal_A'],optimizer=r['optimizer']))
report='''# Current coupling and earlier joint-fit evidence

This audit reads source and existing scalar receipts only. No estimator,
training, numerical runtime, prediction array or target array is executed or
decoded, and no new model or training protocol is selected.

The current fitted P surface is coupled to fixed birth weights. The native
birth-node function consumes the returned GNN state, standardizes its64
channels, multiplies by A[64,4], and applies tanh plus the fixed per-layer
cstar coefficients. The current training translation updates only the11 P
leaves. A, cstar and covariance-edge weights remain frozen, while the source
still feeds the current hs into A(H). No detach is inserted at that use.
This source fact does not certify native gradients or useful optimization.

The64-channel birth-node shape is explicitly checked in native code. Simply
widening the GNN state in its payload would not satisfy that guard or the
A[64,4] interface. Message width is a different dimension: the previous
capacity audit established that it is derived from msg_w and can change
while the state stays64. Neither observation demonstrates compatibility,
cost, resource success or accuracy for an untested model.

Joint P/A learning was actually tested on the older COMP2e13d40a host. Two
matched arms each used20 TRAIN networks,three epochs,60 backwards and15
updates, followed by four sealed CHECK predictions. The joint arm updated
the11 P leaves and256 A entries; the control froze A while retaining its H
dependence. Native CHECK37/39/42/48 compared each role's own complete bills.

Joint/COMP adjusted ratio was1.0009260627183014 with2/4wins; joint/P-only
was1.0044040046045324 with1/4win. The original joint-learning gates failed.
The matched P-only arm's0.9965373078260472 ratio and3/4wins qualified only a
small reused-development component, not native40, independent14, production
or a board result. The score file's top-level passed flag denotes its intact
execution; joint_learning_lead is explicitly false.

Do not conflate this completed joint P/A test with current_bd_p_coadaptation_v1:
the latter was an earlier P-only numerical-qualification attempt with frozen
birth coefficients and stopped before its gradient stages or six-epoch fit.
Also distinguish P/A from the later independent-covariance P/Q pilot, whose
much larger negative result was audited separately. Different names do not
establish different missing experiments; inspect the actual parameter leaves.

These old results do not prove a universal coadaptation limit or forecast the
current fitted candidate. They do rule out calling basic joint P/A tuning an
unexecuted idea. The active retirement-native validation and all original
score gates remain unchanged, with no partial current accuracy preview.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
paths=[BANK,MODEL,FIT/'fit_result.json',score_path,JOINT/'PROTOCOL.md',JOINT/'TRAIN_PROTOCOL.md',
    JOINT/'TRAINING_AND_BUILD_RESULT_20260909.md',JOINT/'CHECK4_RESULT_20260909.md',
    A/'current_bd_p_coadaptation_v1/RESULT_V1.md',D/'joint_covariance_fit_diagnostic_v1/RESULT.md',
    D/'current_p_capacity_history_audit_v1/audit.json',HERE/'current_birth_node_function.py',HERE/'RESULT.md',HERE/'audit.py']
assert not {'numpy','torch','flopscope'} & set(sys.modules)
write(HERE/'audit.json',dict(completed=True,utc=datetime.now(timezone.utc).isoformat(),source_only=True,
    pins=[pin(p) for p in paths],historical_matched_training=training,historical_summary=score['summary'],
    historical_comparisons=score['comparisons'],historical_joint_learning_lead=False,historical_p_only_component=True,
    current_state_width_guard=64,current_birth_A_shape=[64,4],current_fit_frozen=['A','cstar','edge'],
    current_hs_used_without_detach_at_birth_call=True,native_derivative_certified=False,
    estimator_calls=0,new_fit=False,new_model_or_protocol_selected=False,target_arrays_opened=False,
    prediction_arrays_decoded=False,before_current_native40_score=True))
print(dict(completed=True,audit=pin(HERE/'audit.json'),estimator_calls=0),flush=True)
