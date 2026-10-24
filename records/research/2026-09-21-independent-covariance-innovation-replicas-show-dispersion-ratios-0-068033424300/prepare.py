import ast
from common import *
previous=read(D/'continuation_checkpoint_v82.json')
assert previous['checkpoint_version']==82 and not previous['goal_achieved']
for k in ('study','native_study','official_bank'):check(previous[k])
for directory in (FIT,OLD):
    closeout=read(directory/'closeout.json')
    for p in closeout['artifacts']+closeout['external_pins']:check(p)
assert not read(FIT/'result.json')['passed']
receipt=read(FIT/'data_0035.json');assert receipt['rank']==35 and receipt['role']=='TRAIN'
for k in ('arrays','capture','truth','weights','old_statistics'):check(receipt[k])
replica=A/'covariance_replica_truth_v1/rank_0035_8M_v1.npz';rmeta=read(replica.with_suffix('.json'))
assert pin(replica)['sha256']==rmeta['output_sha256']=='e6a7787d974c6b56f39a4444bd54c793ff3efa6e62acb55054774ddb2a2e3358'
assert rmeta['rank']==35 and rmeta['samples']==8_000_000 and rmeta['seed_xor']=='0x4153545241434f56'
assert pin(rmeta['source'])['sha256']==rmeta['source_sha256']
source=(FIT/'common.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tangent_cov')
text=ast.get_source_segment(source,node)+'\n'
with (HERE/'tangent.py').open('x') as f:f.write(text)
with (HERE/'model.py').open('xb') as f:f.write((FIT/'model.py').read_bytes())
write(HERE/'source_copies.json',dict(model=dict(old=pin(FIT/'model.py'),new=pin(HERE/'model.py'),byte_exact=True),
    tangent=dict(old=pin(FIT/'common.py'),new=pin(HERE/'tangent.py'),function='tangent_cov',ast_segment_exact=True)))
meta=dict(rank=35,layers=list(range(8,15)),data=receipt['arrays'],capture=receipt['capture'],truth_A=receipt['truth'],truth_B=pin(replica),
    weights=receipt['weights'],old_statistics=receipt['old_statistics'],data_receipt=pin(FIT/'data_0035.json'),
    models={arm:pin(FIT/f'{arm}_parameters.npz') for arm in ARMS},bank_parameters=pin(FIT/'bank_parameters.npz'),
    reference_B_receipt=pin(replica.with_suffix('.json')),parent=pin(PARENT),rank35_used_in_training=True)
write(HERE/'inputs.json',meta)
paths=[D/'continuation_checkpoint_v82.json',D/'official_best_331684.json',D/'run_guarded.py',PARENT,FIT/'closeout.json',OLD/'closeout.json',
    FIT/'data_0035.json',FIT/'data_0035.npz',FIT/'models_sealed.json',FIT/'bank_parameters.npz',FIT/'common.py',FIT/'model.py',
    FIT/'output_only_parameters.npz',FIT/'nonlinear_parameters.npz',FIT/'output_only_fit.json',FIT/'nonlinear_fit.json',
    FIT/'POST_FIT_SCOPE.md',replica,replica.with_suffix('.json'),rmeta['source'],A/'run_covariance_replica_mc_v1.py',
    A/'COVARIANCE_INNOVATION_REPLICA_V1_VERDICT.md',A/'covariance_innovation_replica_r35_v1.json',
    D/'fp64_sketch_endpoint_v1/RESULT.md',D/'fp64_sketch_endpoint_v1/closeout.json',D/'terminal_covariance_supervision_v1/RESULT.md']
paths += [Path(receipt[k]['path']) for k in ('arrays','capture','truth','weights','old_statistics')]
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(paths)],
    before_new_prediction_or_reference_decode=True,no_fit_or_model_change=True))
print(dict(sealed=True,rank=35,existing_independent_reference=True),flush=True)
