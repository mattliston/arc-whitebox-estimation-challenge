"""Fixed selected-Gram D5 prototype plus two independently screened exact companions."""
from pathlib import Path
from types import SimpleNamespace
import ast, copy, difflib, hashlib, json

D=Path(__file__).resolve().parent; H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=H/'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
GN=H/'hill_climb_s34_gn_pair_reuse_v1'
SCAN=H/'hill_climb_s34_oldtier_scan_elision_v1'
def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    return x.replace(T.parent.name,D.name) if isinstance(x,str) else x

assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
assert pin(GN/'fragment_cached_v2.py')['sha256']=='546974ee1e54bd543c4b3ed389406d4bbaa56dcc11a44deac1f9330647f5ba3d'
assert pin(SCAN/'fragment.py')['sha256']=='1fc78706a4f0fe0a6c59c25e9087e87ce7c2df1abda828359d1089e012aa6b1c'
for d in (GN,SCAN):
    r=json.loads((d/'native2_assessment.json').read_text())
    assert r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['parity_layers']==list(range(16)) and r['configured_parent_bytes_exact']
    assert r['raw_pct']==0 and r['comparator']==pin(P)

source=P.read_text(); fragment=(D/'fragment.py').read_text()
gn=(GN/'fragment_cached_v2.py').read_text();scan=(SCAN/'fragment.py').read_text()
tree=ast.parse(source);lines=source.splitlines(True)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'
       and any(isinstance(x,ast.Name) and x.id=='_need_core' for x in ast.walk(n))]
assert len(nodes)==1; node=nodes[0]
start=sum(map(len,lines[:node.lineno-1]));end=sum(map(len,lines[:node.end_lineno]))
old='Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))'
new='Y = _ng_selected_gram(J, weight, _jn_rows)'
body=source[start:end];assert body.count(old)==1
patched=source[:start]+body.replace(old,new,1)+source[end:]
assert patched.count(new)==1 and patched.replace(new,old,1)==source
changed=[a.name for a,b in zip(tree.body,ast.parse(patched).body) if ast.dump(a)!=ast.dump(b)]
assert changed==['_shared_old384_push']
candidate=patched+'\n'+fragment+'\n'+gn+'\n'+scan
assert candidate[:-len('\n'+fragment+'\n'+gn+'\n'+scan)].replace(new,old,1)==source
compile(candidate,str(D/'candidate.py'),'exec')

# Source-only public-shape proof: all added views use existing arena partitions.
ns=dict(_IPT_REGION={5:(137682944,137682944,68841472),6:(90354432,120472576,120472576)},
    _SCRATCH=(SimpleNamespace(shape=(391602176,)),),_KS_C5=18.36,_KS_C6=30.60,
    _KS_PAD=1.35,_KS_CAP=1<<20,_KS_PRICE=750000.,_KS_REQ5=72,_KS_PLANS={})
names={'_ipt_fits','_p6_bill','_p6_secs','_ks_plan'}
defs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
exec(compile(ast.Module(body=list(defs.values()),type_ignores=[]),'scalar-pricing','exec'),ns)
bind={'tuple':tuple}
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('_PB_SINGLE','_KS_ACC_MN') for x in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'scalar-binding','exec'),{'__builtins__':{},'tuple':tuple},bind)
keys=ast.literal_eval(next(n.value for n in ast.parse(fragment).body if isinstance(n,ast.Assign)
    and any(isinstance(x,ast.Name) and x.id=='_NG_PREBIND' for x in n.targets)))
assert len(keys)==23 and not set(keys).intersection(bind['_PB_SINGLE'])
assert len(bind['_PB_SINGLE'])==649 and (576,384) in bind['_KS_ACC_MN']
for kind,L,rows,cols in keys:
    region=ns['_IPT_REGION'][L]
    assert rows%(1<<L)==cols%(1<<L)==0
    assert 7**L*(rows//(1<<L))*(cols//(1<<L))<=region[kind]

bill=ns['_p6_bill'];secs=ns['_p6_secs'];records=[]
def ks(m,k,n):
    M,N,qs=ns['_ks_plan'](m,k,n)
    assert (M,N)==(m,n)
    return (sum(bill(6,M,q,N,0) for q in qs)-(len(qs)-1)*58.88*M*N
        +(len(qs)-1)*7**6*(M//64)*(N//64),sum(secs(6,M,q,N,0) for q in qs))
for m,k,r in ((960,6144,384),(832,5568,320),(832,4672,384)):
    shapes=[(min(128,r-s),k,r-s) for s in (0,128,256)]
    assert all(ns['_ipt_fits'](5,*shape) for shape in shapes)
    oldbill,oldtime=ks(m,k,r);non,nt=ks(m-r,k,r)
    extra=(m-r)*k+r*k+k*sum(r-s for s in (0,128,256))+6*r*r+4*m*r+10*m
    newbill=non+sum(bill(5,*shape,0) for shape in shapes)+extra
    records.append(dict(shape=[m,k,r],strips=shapes,old_model_flops=oldbill,
        new_model_flops=newbill,extra_gather_copy_allowance=extra,saved_flops=oldbill-newbill,
        full_transform_seconds_delta=nt+sum(secs(5,*shape,0) for shape in shapes)-oldtime))
saved=sum(r['saved_flops'] for r in records)
price=dict(records=records,forecast_saved_flops=saved,reference_s34_rank35_flops=267405124808,
    forecast_pct=100*saved/267405124808,
    full_transform_seconds_delta=sum(r['full_transform_seconds_delta'] for r in records),
    shape_evidence=pin(H/'hill_climb_s32_bill_census_v1/census.json'),
    scope='Three saved S32 join shapes, current S34 kernel formulas; forecast only. No claim of current per-call measurement or runtime qualification.',
    caveats='Includes recurring gather/copy allowance. Adds23 existing-arena setup keys; actual setup and unseen-shape lazy-binding bill must be measured. No automatic .42% saving claim.')
assert price['forecast_pct']>=.1
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P),tofile=str(D/'candidate.py'))))
once(D/'price.json',price)
protocol=dict(schema='s34-nystrom-blocksym-gn-scan-v1',root_selected=True,parent=pin(P),
    change='Only new-J Nystrom contribution: nonselected rows use existing mm routing; selected weighted Gram uses three D5 upper strips/full diagonal blocks and paid mirrored copies. Old Qp/core term is source-identical.',
    formula='G = J_selected diag(weight) J_selected.T; nonselected = J_other diag(weight) J_selected.T.',
    rounding='Real-arithmetic identity; D5 and symmetry copying alter FP32 rounding and may affect Nystrom conditioning. Original approximate-candidate gates apply.',
    companions=[dict(fragment=pin(GN/'fragment_cached_v2.py'),native2=pin(GN/'native2_assessment.json'),protocol=pin(GN/'protocol.json')),
                dict(fragment=pin(SCAN/'fragment.py'),native2=pin(SCAN/'native2_assessment.json'),protocol=pin(SCAN/'protocol.json'))],
    companion_semantics='GN reuses identical current pair bonds. Scan elision preserves dtype but removes intermediate isfinite/all early exceptions; final all16 finite checks and all other source guards remain.',
    no_isolated_nystrom_gain_claim=True,parity_layers=list(range(7)),first_changed_join=7,
    arena_size_unchanged=True,tr_cache_cleared_before_and_after_each_forced_product=True,
    prebind_keys=keys,setup_counts=dict(bound=705,lvfuse=672,psdb_fused=9),price=pin(D/'price.json'),
    source_inverse_exact=True,changed_parent_functions=changed,all_value_operations_paid=True,
    roster=[35,100],scope='native2_only',fit_or_sweep=False,boundary_authorized=False,
    broad_qualified=False,serving_qualified=False,submitted=False,model_executed=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),
    fragment=pin(D/'fragment.py'),companions=protocol['companions'],protocol=pin(D/'protocol.json'),
    diff=pin(D/'source_diff.patch'),price=pin(D/'price.json'),source_inverse_exact=True,
    exact_parent_prefix=False,changed_parent_functions=changed,old_statement=old,new_statement=new,
    numpy_torch_or_model_executed=False))
base=json.loads(T.read_text());c=rewrite(copy.deepcopy(base))
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(7))
c['arms']['candidate']['setup']['counts']=protocol['setup_counts']
c['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'fragment.py',D/'build.json',D/'protocol.json',D/'price.json',D/'source_diff.patch',
    GN/'fragment_cached_v2.py',GN/'native2_assessment.json',SCAN/'fragment.py',SCAN/'native2_assessment.json',H/'official_bank_S34_v1.json')]
c['interpretation']='One fixed Nyström upper-strip D5 implementation plus independently screened exact GN/scan companions on S34. Source inverse and L0–6 byte parity; original .1% N2 gates. No isolated-component-gain claim, no broader authority.'
assert c['arms']['base']==base['arms']['base'] and c['arms']['parity_parent']==base['arms']['parity_parent']
assert c['prospective_scope']=='native2_only'
assert all(c['stages'][s][k]==base['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
once(D/'native_screen.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),build=pin(D/'build.json'),protocol=pin(D/'protocol.json'),forecast_pct=price['forecast_pct'])))
