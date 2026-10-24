from pathlib import Path
import ast,copy,json,hashlib,math
import numpy as np
HERE=Path(__file__).resolve().parent;H=HERE.parent
PARENT=H/'classical_covariance_response_v1';PROBE=H/'native_innovation_readout_response_v1';BASE=H/'recurrent_covariance_observer_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for s in ('seal.json','verification_seal.json'):
    for e in read(HERE/s)['sources']:assert pin(e['path'])==e
original=(PARENT/'candidate.py').read_text();current=(HERE/'candidate.py').read_text()
marker='\n\n# Fixed late numerical training reference.\n';assert current.count(marker)==1
prefix,footer=current.split(marker)
def body(source):
    lines=source.splitlines(keepends=True)
    ns=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'
        and '_slr_require(_JOIN_NYSTROM' in ''.join(lines[n.lineno-1:n.end_lineno])]
    assert len(ns)==1;n=ns[0]
    return n,''.join(lines[:n.lineno-1]),''.join(lines[n.end_lineno:])
old,pre0,post0=body(original);new,pre1,post1=body(prefix)
assert pre0==pre1 and post0==post1
test=ast.dump(ast.parse('_SR_ON and layer >= 9',mode='eval').body);changes=[]
class Undo(ast.NodeTransformer):
    def visit_If(self,node):
        if ast.dump(node.test)==test:
            assert len(node.body)==1 and isinstance(node.body[0],ast.Assign)
            a=node.body[0];target=a.targets[0].id
            if target=='Y':
                assert isinstance(a.value,ast.Call) and a.value.func.id=='_sr_wide_sketch' and node.orelse
                changes.append('sketch');return node.orelse
            assert target=='Qn' and not node.orelse
            assert ast.dump(a.value)==ast.dump(ast.parse('Qn.astype(fnp.float32)',mode='eval').body)
            changes.append('Q_cast');return []
        return self.generic_visit(node)
restored=Undo().visit(copy.deepcopy(new));assert changes==['sketch','Q_cast'] and ast.dump(restored)==ast.dump(old)
assert read(HERE/'seal.json')['source_inverse_exact']
prep=read(HERE/'preparation.json');assert prep['passed'] and len(prep['rows'])==10 and all(r['bytes_exact'] for r in prep['rows'])
result=read(HERE/'result.json');d0=read(HERE/'direction0_result.json')
assert result['directions']==([0,7] if d0['passed'] else [0])
assert not result['final_targets_read'] and not result['probe_accuracy_scored'] and not result['coefficients_fitted']
names=['disabled','zero']+[f'd{d}_{s}' for d in result['directions'] for s in ('plus','minus','halfplus','halfminus')]
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
control={r['mini']:r for r in read(PARENT/'zero/complete.json')['rows']}
with np.load(PROBE/'basis.npz') as z:directions=z['directions'].copy()
packets={};bills=[]
for gpath in HERE.glob('*_guard.json'):
    if gpath.name=='verification_guard.json':continue
    g=read(gpath);assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
for name in names:
    r=read(HERE/name/'complete.json');assert r['first9_exact'] and r['training_reference_only'] and not r['final_targets_read']
    disabled=name=='disabled';assert r['covariance_counts']==dict(full=12,diagonal=2)
    assert r['reference_counts']==dict(late_dispatch=0 if disabled else 12,wide_sketch=0 if disabled else 4)
    packets[name]={}
    prior_name='zero' if disabled else name;fast={x['mini']:x for x in read(PROBE/prior_name/'complete.json')['rows']}
    if name in ('disabled','zero'):expected=np.zeros((1,32),dtype='float32')
    else:
        d=int(name[1]);amp={'plus':.125,'minus':-.125,'halfplus':.0625,'halfminus':-.0625}[name.split('_')[1]]
        expected=(amp*directions[:,d]).astype('float32')[None,:]
    for row in r['rows']:
        mini=row['mini'];e=row['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:a={k:z[k].copy() for k in z.files}
        assert a['readout'].tobytes()==expected.tobytes() and row['switch_restored']
        e=base[mini]['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:baseline=z['prediction'].copy()
        assert a['prediction'][:9].tobytes()==baseline[:9].tobytes()
        if disabled:
            e=control[mini]['packet'];assert pin(e['path'])==e
            with np.load(e['path']) as z:assert a['prediction'].tobytes()==z['prediction'].tobytes()
            assert row['flops']==control[mini]['flops']
        assert row['fast_parent_flops']==fast[mini]['flops'] and row['covariance_baseline_flops']==control[mini]['flops']
        assert row['fast_bill_ratio']==row['flops']/row['fast_parent_flops']
        assert row['covariance_baseline_bill_ratio']==row['flops']/row['covariance_baseline_flops']
        assert np.isfinite(a['prediction']).all() and np.min(a['prediction'])>=0
        packets[name][mini]=a;bills.append(dict(name=name,mini=mini,fast_ratio=row['fast_bill_ratio'],covariance_ratio=row['covariance_baseline_bill_ratio']))
def metric(z,p,m,ph,mh):
    rh=[(float(a)-float(b))/.25 for a,b in zip(p,m)];rl=[(float(a)-float(b))/.125 for a,b in zip(ph,mh)]
    sq=lambda x:math.fsum(v*v for v in x)
    secant=math.sqrt(sq([a-b for a,b in zip(rh,rl)])/max(sq(rl),1e-60))
    rms=math.sqrt(sq([.125*v for v in rl])/len(z))
    curvature=math.sqrt(sq([float(a)+float(b)-2*float(c) for a,b,c in zip(p,m,z)])/max(sq([float(a)-float(b) for a,b in zip(p,m)]),1e-60))
    return secant,rms,curvature
checked=0;passed=[]
for row in result['rows']:
    d,mini=row['direction'],row['mini'];p=[packets[f'd{d}_{s}'][mini] for s in ('plus','minus','halfplus','halfminus')]
    for layer,r in row['layers'].items():
        l=int(layer);v=metric(packets['zero'][mini]['prediction'][l],*[x['prediction'][l] for x in p])
        for key,x in zip(('secant_relative','response_rms','curvature'),v):assert math.isclose(x,r[key],rel_tol=1e-13,abs_tol=1e-15)
        good=v[0]<=.05 and v[1]>=5e-7 and v[2]<=.10
        assert good==r['passed'];checked+=1
        if l==15:assert row['passed']==good;passed.append(good)
assert result['passed']==all(passed)
once(HERE/'verification.json',dict(passed=True,reference_qualification_passed=result['passed'],source_ast_inverse_exact=True,
    unchanged_source_segments_byte_exact=True,all_readouts_and_prefixes_verified=True,scalar_layer_checks=checked,
    bills=bills,result=pin(HERE/'result.json'),final_targets_read=False,coefficients_fitted=False))
print(json.dumps(dict(verification_passed=True,reference_qualification_passed=result['passed'],scalar_layer_checks=checked,bills=bills)),flush=True)
