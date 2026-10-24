import ast
from common import *
cp=D/'continuation_checkpoint_v86.json';assert sha(cp)=='fc2ff83cff7d703b26075d8278b83d0edad0f31aa895e6f6bf3ea83e818181e3'
old=read(cp)
for k in ('study','workers','official_bank','pending_release','pending_candidate','pending_archive','rejection'):verify(old[k])
assert sha(PARENT)==PARENT_SHA
assert sha(MODEL)=='03fc240cf11f6b4246f6f644aa1447d93a962fc2ed343df8ce6be2623ff60776'
assert sha(P0)=='166b3c67ec4d27793d433e38e15086b95d052426fdf435a79067cc07a2a4dd7e'
verdict=read(A/'learned_pair_history_verdict_v1.json');receipt=read(A/'learned_pair_history_rows_v1.json')
assert not verdict['information_gate_pass'] and verdict['epochs']==100 and receipt['passed']
assert sha(A/'train_learned_pair_history_v1.py')==verdict['source_sha256']
assert sha(A/'train_covariance_innovation_probe_v1.py')==verdict['original_trainer_sha256']
for name,value in receipt['source_hashes'].items():assert sha(A/name)==value,name
prior_inputs=D/'current_edge_nonlinear_native_v1/inputs.json';meta=read(prior_inputs)
external=[cp,PARENT,MODEL,P0,D/'run_guarded.py',prior_inputs,D/'official_best_331684.json',
    A/'learned_pair_history_verdict_v1.json',A/'learned_pair_history_rows_v1.json',
    A/'LEARNED_PAIR_HISTORY_INFORMATION_DESIGN_V1.md',A/'LEARNED_PAIR_HISTORY_INFORMATION_V1_CLOSED.md',
    A/'train_learned_pair_history_v1.py',A/'train_covariance_innovation_probe_v1.py',
    A/'generate_learned_pair_history_row_v1.py',A/'generate_covariance_innovation_probe_v1.py',
    A/'radial_features.py',A/'radial_gnn_numpy.py',D/'terminal_latent_covariance_information_v3/NEGATIVE_RESULT.md',
    D/'current_edge_nonlinear_panel_v1/RESULT.md']
verify(meta['bank_triple']);external.append(Path(meta['bank_triple']['path']))
for p in meta['weights'].values():verify(p);external.append(Path(p['path']))
for p in meta['truths'].values():verify(p);external.append(Path(p['path']))
for row in meta['parent'].values():
    for key in ('prediction','receipt','source'):verify(row[key]);external.append(Path(row[key]['path']))
for r in read(meta['bank_triple']['path'])['records']:
    assert sha(r['output'])==r['output_sha256'];external.append(Path(r['output']))
legacy=[]
for rank in (400,500):
    row=next(x for x in receipt['records'] if x['rank']==rank)
    hp=A/f'rows_learned_pair_history_v1/row_{rank:04d}/history.npy'
    dp=A/f'rows_covariance_innovation_probe_v2/row_{rank:04d}/data.npz'
    assert sha(hp)==row['history_sha256'] and sha(dp)==row['original_row_sha256']
    legacy.append(dict(rank=rank,history=pin(hp),data=pin(dp),expected=verdict['final_valid'][str(rank)]))
    external += [hp,dp]
meta.update(legacy=legacy,parent_source=pin(PARENT),history_model=pin(MODEL),history_P0=pin(P0),
    old_proxy_gate_preserved=False if verdict['information_gate_pass'] else True,
    model_current_hash_not_previously_sealed=True,ordinary_1B_targets_not_decoded=True)
write(HERE/'inputs.json',meta)
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+
    [pin(p) for p in dict.fromkeys(external)],before_tensor_decode=True,before_current_native_calls=True,
    before_ordinary_1B_targets=True,original_proxy_gate_still_failed=True,stopping_decision_explicitly_revised=True))
print(dict(sealed=True,model_bytes=MODEL.stat().st_size,no_arrays_decoded=True),flush=True)
