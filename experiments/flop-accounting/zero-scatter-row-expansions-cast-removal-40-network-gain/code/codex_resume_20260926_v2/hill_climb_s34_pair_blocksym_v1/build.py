"""S34 packed symmetric pair-series; source/scalar checks only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json

D=Path(__file__).resolve().parent; H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=H/'hill_climb_s34_terminal_full_star_v1/native_screen.json'
def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    if isinstance(x,str):return x.replace(T.parent.name,D.name)
    return x

assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='be0a6a520dfe81bcc49c7fdcc5bc9baacf9ce90f762c5b4a89cbfa735680fd76'
source=P.read_text(); fragment=(D/'fragment.py').read_text()
start="    Z0, Z1, Z2, Z3, Z4 = st['Zv']\n"
end="    m2 = _ps32(2 * m)\n"
assert source.count(start)==source.count(end)==1
a=source.index(start); b=source.index(end,a)
old=source[a:b]
new=("    if n == 1024 and L.get('psp_blocks') == 128:\n"
     "        cov, pk21, pk22 = fnp.unstack(_e2_H(_psp_pair_series(st, n, _e2_H)) * fnp.stack((_ps_ss32, _ps_vs32, _ps_vv32)))\n"
     "    else:\n"+''.join('    '+line if line.strip() else line for line in old.splitlines(True)))
capture_old="            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate)"
capture_new=capture_old+"\n            _E2_LAYER['psp_blocks'] = (128 if l > 0 and cov32 and _FULL_LOWER_MOMENTS and n == 1024 and _AB_ON and _WINOGRAD and _psdb_fits() and all(_ipt_fits(6, pm, pk, pn) for pm, pk, pn in _PSD_SHAPES) else 0)"
assert source.count(capture_old)==1
patched=source[:a]+new+source[b:]
patched=patched.replace(capture_old,capture_new,1)
assert patched.replace(capture_new,capture_old,1).replace(new,old,1)==source
trees=(ast.parse(source),ast.parse(patched))
changed=[(a,b) for a,b in zip(trees[0].body,trees[1].body) if ast.dump(a)!=ast.dump(b)]
assert len(trees[0].body)==len(trees[1].body)
assert [a.name for a,b in changed]==['_nonlinear_pair_batch','_predict_binary']
assert all(a.name==b.name for a,b in changed)
assert "_PSDB_GROUPS = ((0, 128, 256), (128, 128, 128))" in source
assert "cov = _sym_upper(wt, first)" in source and "_sym_upper = _psdb_sym_upper" in source
candidate=patched+'\n\n'+fragment
compile(candidate,str(D/'candidate.py'),'exec')

# Exhaustive scalar block-index ownership proof, no array library or model.
n=1024; block=128; nb=n//block
pairs=[(i,i) for i in range(nb)]+[(i,j) for i in range(nb) for j in range(i+1,nb)]
covered={}; mirrors=0
for i,j in pairs:
    assert (i,j) not in covered;covered[i,j]='computed'
    if i!=j:assert (j,i) not in covered;covered[j,i]='mirrored';mirrors+=1
assert len(pairs)==36 and mirrors==28 and len(covered)==64
assert all(covered[i,i]=='computed' for i in range(nb))
nn=n*n; q=len(pairs)*block*block
# Scalar conservative cost, including overallowance for paid indexing/views.
gross=14*63*nn
extra=14*25*nn+72*nn # all recurring gathers/scatters plus cold view allowance
mean_bill=267228330900.25
price=dict(gross_saved_flops=gross,conservative_extra_flops=extra,
           forecast_saved_flops=gross-extra,reference_native8_mean_flops=mean_bill,
           forecast_saving_pct=100*(gross-extra)/mean_bill,
           explanation='216 n^2/layer original ordered series; BA plus 2*36/64 packed channels saves63 n^2. Overallow25 n^2/layer for paid take/getitem/scatter and72 n^2 once for shape views. Model forecast only; actual native bill decides.',
           local_cost_evidence={'take':'numel(output)','getitem':'4*numel(output)','put':'numel(indices)','copyto':'numel(destination)','reshape':'numel(input)'})
assert price['forecast_saving_pct']>=.1
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P),tofile=str(D/'candidate.py'))))
once(D/'price.json',price)
once(D/'protocol.json',dict(schema='s34-pair-blocksym-v1',root_selected=True,parent=pin(P),
    change='AA/BB raw ordered pair series on36 upper/full-diagonal128blocks, mirror28offblocks; BA full. Full FT/P16 and unchanged central moments preserved.',
    known_symmetric_scope='Only live full-lower cov32 path with current D5 symmetry dispatch guards. Cpre/k22 may remain asymmetric within each128diagonalblock; those blocks are computed in full.',
    arbitrary_K21_and_T31_supported=True,parity_layers=[0],expected_arithmetic='Same per-element multiply/multiply/reduce, term-addition/division/ordered-final-sum sequence.',
    original_source_inverse_exact=True,changed_functions=['_nonlinear_pair_batch','_predict_binary'],
    added_source=pin(D/'fragment.py'),block_index_scalar_proof=dict(computed=36,mirrored=28,total=64,full_diagonal=8),
    index_constants_only=True,all_dynamic_values_gathered_fresh=True,gn_reuse_not_included=True,
    arena_size_unchanged=True,registered_cache_lifecycle=True,setup_counts=dict(bound=683,lvfuse=649,psdb_fused=9,pair_block_indices=1),
    cost=pin(D/'price.json'),scope='native2_only',roster=[35,100],original_gates_unchanged=True,
    broad_qualified=False,serving_qualified=False,submitted=False,model_executed=False))
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),fragment=pin(D/'fragment.py'),
    primitive=pin(D/'probe.py'),
    protocol=pin(D/'protocol.json'),diff=pin(D/'source_diff.patch'),price=pin(D/'price.json'),
    source_inverse_exact=True,changed_functions=['_nonlinear_pair_batch','_predict_binary'],
    appended_functions=[x.name for x in ast.parse(fragment).body if isinstance(x,ast.FunctionDef)],
    arrays_imported=False,model_executed=False))
original=json.loads(T.read_text()); c=rewrite(copy.deepcopy(original))
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=[0]
c['arms']['candidate']['setup']['counts']=dict(bound=683,lvfuse=649,psdb_fused=9,pair_block_indices=1)
c['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'fragment.py',D/'probe.py',D/'build.json',D/'protocol.json',D/'price.json',D/'source_diff.patch',H/'official_bank_S34_v1.json')]
c['interpretation']='One fixed symmetry implementation on S34; full128diagonal blocks, asymmetric BA and downstream central moments preserved. New paid shape indices only. Native2 controls/gates unchanged; no GN-reuse component or broader qualification.'
assert c['arms']['base']==original['arms']['base'] and c['arms']['parity_parent']==original['arms']['parity_parent']
assert all(c['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
once(D/'native_screen.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),build=pin(D/'build.json'),protocol=pin(D/'protocol.json'),forecast=price)))
