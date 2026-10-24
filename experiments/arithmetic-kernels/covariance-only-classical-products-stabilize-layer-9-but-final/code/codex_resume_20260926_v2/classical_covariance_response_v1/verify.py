from pathlib import Path
import ast,copy,json,hashlib,math
import numpy as np
HERE=Path(__file__).resolve().parent;H=HERE.parent
PARENT=H/'native_covariance_innovation_fit_v1/candidate.py'
PROBE=H/'native_innovation_readout_response_v1'
BASE=H/'recurrent_covariance_observer_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for seal in ('seal.json','verification_seal.json'):
    for e in read(HERE/seal)['sources']:assert pin(e['path'])==e
original=PARENT.read_text();current=(HERE/'candidate.py').read_text()
marker='\n\n# Fixed covariance-only arithmetic intervention.\n'
assert current.count(marker)==1
prefix,footer=current.split(marker)
edits=[('Cpre, _bb_star_pre = _cr_dispatch(l,w32,C,star)','Cpre, _bb_star_pre = _flms_cov_star(w32, C, star)'),
    ('v, active_v = (_cr_diagonal_dispatch(l,w32,C) if _FULL_LOWER_MOMENTS','v, active_v = (_flms_covariance_diagonal(w32, C) if _FULL_LOWER_MOMENTS')]
for a,b in edits:assert prefix.count(a)==1;prefix=prefix.replace(a,b)
assert prefix==original
old={n.name:n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
new={n.name:n for n in ast.parse(footer).body if isinstance(n,ast.FunctionDef)}
expected_cov=copy.deepcopy(old['_flms_cov_star']);expected_cov.name='_cr_cov_star'
firsts=0;syms=0
for n in expected_cov.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
        if n.targets[0].id=='first':
            assert ast.dump(n.value)==ast.dump(ast.parse('_mm(C32,w)',mode='eval').body)
            n.value=ast.BinOp(left=ast.Name(id='C32',ctx=ast.Load()),op=ast.MatMult(),right=ast.Name(id='w',ctx=ast.Load()));firsts+=1
        if n.targets[0].id=='cov':
            assert isinstance(n.value,ast.Call) and n.value.func.id=='_sym_upper'
            n.value.func.id='_cr_sym_upper';syms+=1
assert firsts==syms==1 and ast.dump(expected_cov)==ast.dump(new['_cr_cov_star'])
expected_diag=copy.deepcopy(old['_flms_covariance_diagonal']);expected_diag.name='_cr_cov_diagonal'
changed=0
for n in expected_diag.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='pushed':
        assert isinstance(n.value,ast.Call) and n.value.func.id=='_mm' and len(n.value.args)==2
        n.value=ast.BinOp(left=n.value.args[0],op=ast.MatMult(),right=n.value.args[1]);changed+=1
assert changed==1 and ast.dump(expected_diag)==ast.dump(new['_cr_cov_diagonal'])
result=read(HERE/'result.json');d0=read(HERE/'direction0_result.json')
assert not result['final_targets_read'] and not result['probe_accuracy_scored'] and not result['coefficients_fitted']
assert result['directions']==([0,7] if d0['passed'] else [0])
names=['disabled','zero']+[f'd{d}_{s}' for d in result['directions'] for s in ('plus','minus','halfplus','halfminus')]
with np.load(PROBE/'basis.npz') as z:directions=z['directions'].copy()
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
packets={};bills=[]
for name in names:
    g=read(HERE/(name+'_guard.json'));assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
    r=read(HERE/name/'complete.json');assert r['first9_exact'] and r['covariance_only_replacement'] and not r['final_targets_read']
    parent_name='zero' if name=='disabled' else name
    prior={x['mini']:x for x in read(PROBE/parent_name/'complete.json')['rows']}
    packets[name]={}
    if name in ('zero','disabled'):expected=np.zeros((1,32),dtype='float32')
    else:
        d=int(name[1]);amp={'plus':.125,'minus':-.125,'halfplus':.0625,'halfminus':-.0625}[name.split('_')[1]]
        expected=(amp*directions[:,d]).astype('float32')[None,:]
    for row in r['rows']:
        mini=row['mini'];e=row['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:a={k:z[k].copy() for k in z.files}
        assert a['readout'].tobytes()==expected.tobytes()
        e=base[mini]['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:baseline=z['prediction'].copy()
        assert a['prediction'][:9].tobytes()==baseline[:9].tobytes()
        assert np.isfinite(a['prediction']).all() and np.min(a['prediction'])>=0
        if name=='disabled':assert a['prediction'].tobytes()==baseline.tobytes() and row['flops']==prior[mini]['flops']
        assert row['parent_flops']==prior[mini]['flops'] and row['bill_ratio']==row['flops']/row['parent_flops']
        if not name.startswith('d7'):assert all(row['trace_checks'].values())
        packets[name][mini]=a;bills.append(dict(name=name,mini=mini,ratio=row['bill_ratio']))
checked=0;passes=[]
def metric(z,p,m,ph,mh):
    r1=[(float(a)-float(b))/.25 for a,b in zip(p,m)]
    r2=[(float(a)-float(b))/.125 for a,b in zip(ph,mh)]
    sq=lambda v:math.fsum(x*x for x in v)
    secant=math.sqrt(sq([a-b for a,b in zip(r1,r2)])/max(sq(r2),1e-60))
    response=math.sqrt(sq([.125*x for x in r2])/len(z))
    curvature=math.sqrt(sq([float(a)+float(b)-2*float(c) for a,b,c in zip(p,m,z)])/max(sq([float(a)-float(b) for a,b in zip(p,m)]),1e-60))
    return secant,response,curvature
for row in result['rows']:
    d,mini=row['direction'],row['mini'];others=[packets[f'd{d}_{s}'][mini] for s in ('plus','minus','halfplus','halfminus')]
    for layer,saved in row['layers'].items():
        l=int(layer);values=metric(packets['zero'][mini]['prediction'][l],*[a['prediction'][l] for a in others])
        for key,v in zip(('secant_relative','response_rms','curvature'),values):assert math.isclose(v,saved[key],rel_tol=1e-13,abs_tol=1e-15)
        passed=values[0]<=.05 and values[1]>=5e-7 and values[2]<=.1
        assert saved['passed']==passed;checked+=1
        if l==15:assert row['passed']==passed;passes.append(passed)
assert result['passed']==all(passes)
once(HERE/'verification.json',dict(passed=True,native_qualification_passed=result['passed'],source_inverse_exact=True,
    independent_helper_ast_checks=True,all_readouts_and_prefixes_verified=True,scalar_layer_checks=checked,
    complete_bill_ratios=bills,result=pin(HERE/'result.json'),final_targets_read=False))
print(json.dumps(dict(verification_passed=True,native_qualification_passed=result['passed'],scalar_layer_checks=checked,bills=bills)),flush=True)
