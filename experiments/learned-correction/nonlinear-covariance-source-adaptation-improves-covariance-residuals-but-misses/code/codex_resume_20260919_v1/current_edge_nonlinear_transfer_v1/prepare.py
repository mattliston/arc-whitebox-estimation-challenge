import ast
from common import *
assert sha(PARENT)=='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
assert sha(D/'continuation_checkpoint_v80.json')=='6c625bc28321f25eb60db5d22ab443fa0ba099063cbe583bcda495586217b830'
previous=read(D/'continuation_checkpoint_v80.json');check(previous['study']);check(previous['official_bank'])
old=read(OLD/'closeout.json');assert old['completed']
for p in old['artifacts']+old['external_pins']:check(p)
def function(path,name):
    s=path.read_text();t=ast.parse(s)
    return ast.get_source_segment(s,next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name))
assert function(HERE/'common.py','tangent_cov')==function(OLD/'common.py','tangent_cov')
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
external=[D/'continuation_checkpoint_v80.json',D/'official_best_331684.json',PARENT,D/'run_guarded.py',
    OLD/'closeout.json',OLD/'PROTOCOL.md',OLD/'SOURCE_REVIEW.md',OLD/'RESULT.md',OLD/'common.py',OLD/'statistics.py',
    OLD/'fit.py',OLD/'result.json',OLD/'captures_sealed.json',OLD/'wide_qualification.json',OLD/'audit_result.json',
    D/'post_full_lower_scope_v1/SCOPE.md',A/'train_joint_edge_covariance_v1.py',A/'THIRD_RECENTER_ROUND_V1_CLOSED.md']
for row in read(OLD/'captures_sealed.json')['rows']:
    check(row['capture']);external.append(Path(row['capture']['path']))
for rank in TRAIN+VALID:
    external.extend([OLD/f'statistics_{rank:04d}.npz',OLD/f'statistics_{rank:04d}.json',weights(rank),truth(rank)])
write(HERE/'build.json',dict(previous_checkpoint=pin(D/'continuation_checkpoint_v80.json'),old_study=pin(OLD/'closeout.json'),
    tangent_function_byte_identical=True,old_training_parameters=16,new_training_parameters=528,
    matched_output_only_control=True,source_arrays_not_decoded=True))
pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)]
write(HERE/'sources_sealed.json',dict(pins=pins,train=list(TRAIN),valid=list(VALID),before_new_numeric_work=True,
    parent=pin(PARENT),prior_failed_fit_preserved=True))
print(dict(sealed=True,files=len(pins),tangent_function_byte_identical=True),flush=True)
