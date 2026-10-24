"""Fixed retained packed-pair + READ7 + exact scan-elision combination; stdlib only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json
D=Path(__file__).resolve().parent; H=D.parent
PP=H/'hill_climb_s34_pair_blocksym_v1';R7=H/'hill_climb_s34_read7_v1'
SCAN=H/'hill_climb_s34_oldtier_scan_elision_v1'
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=PP/'native_screen.json'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    if isinstance(x,str):return x.replace(PP.name,D.name)
    return x
def checkpin(x):assert pin(x['path'])==x;return x

assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(PP/'candidate.py')['sha256']=='4e82cf45bd46e114a756d20220332f36d20db2bdb2767d95dcae1401d12f337a'
assert pin(PP/'retained_component.json')['sha256']=='354b9e07ce8d5041f6e20a47991f45aa8b3b107f24e7ae126ad41b6ef45c969f'
assert pin(R7/'retained_component.json')['sha256']=='ad0e0e3e710de25bef877d5eb1efdc1da838086b436373bf78e3deeebf279843'
assert pin(R7/'fragment.py')['sha256']=='af467decdcac5ac8e005248984cd10220977df5ac75c27f660f5ec173932ca1e'
assert pin(SCAN/'fragment.py')['sha256']=='1fc78706a4f0fe0a6c59c25e9087e87ce7c2df1abda828359d1089e012aa6b1c'
retained=[]
for folder in (PP,R7):
    receipt=read(folder/'retained_component.json')
    assert receipt['status']=='retained_for_combination' and receipt['native8_runtime_recheck']['passed']
    assert receipt['candidate']==pin(folder/'candidate.py') and receipt['comparator']==pin(P)
    native=read(checkpin(receipt['native8'])['path']);boundary=read(checkpin(receipt['boundary'])['path'])
    assert native['admitted'] and native['passed_integrity'] and all(native['gates'].values())
    assert boundary['passed'] and boundary['receipt_checks_passed'] and all(boundary['gates'].values())
    retained.append(dict(receipt=pin(folder/'retained_component.json'),native8=receipt['native8'],boundary=receipt['boundary'],candidate=receipt['candidate']))
scan=read(SCAN/'native2_assessment.json')
assert scan['admitted'] and scan['passed_integrity'] and all(scan['gates'].values())
assert scan['raw_pct']==0 and scan['parity_layers']==list(range(16)) and scan['configured_parent_bytes_exact']
assert scan['comparator']==pin(P)

parent=P.read_text();pp=(PP/'candidate.py').read_text();r7=(R7/'fragment.py').read_text();sf=(SCAN/'fragment.py').read_text()
inverse=read(PP/'source_inverse.json');checkpin(inverse['parent']);checkpin(inverse['candidate']);checkpin(inverse['fragment'])
suffix=inverse['suffix_separator']+Path(inverse['fragment']['path']).read_text()
assert pp.endswith(suffix);peeled=pp[:-len(suffix)]
for edit in reversed(inverse['edits']):assert peeled.count(edit['after'])==1;peeled=peeled.replace(edit['after'],edit['before'],1)
assert peeled==parent
assert (R7/'candidate.py').read_text()==parent+'\n'+r7
candidate=pp+'\n'+r7+'\n'+sf
assert '_GNPR_PARENT_STATE' not in candidate and '_ng_selected_gram' not in candidate
assert candidate[:-len('\n'+r7+'\n'+sf)]==pp
compile(candidate,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(parent.splitlines(True),candidate.splitlines(True),fromfile=str(P),tofile=str(D/'candidate.py'))))
protocol=dict(schema='s34-packed-pair-read7-scan-v1',root_selected=True,parent=pin(P),retained_components=retained,
    scan_component=dict(fragment=pin(SCAN/'fragment.py'),native2=pin(SCAN/'native2_assessment.json')),
    scope='One combined native2, own boundary, fixed6/native8 and later independent broad validation only on passed gates.',
    component_qualifications='PP and READ7 each passed own N8, boundary and S32 runtime recheck. Scan has isolated exact N2 evidence and qualifies further in the combination.',
    source_inverse=pin(PP/'source_inverse.json'),append_order=['packed_pair','READ7','scan_elision'],
    gn_pair_bond_reuse_excluded=True,nystrom_excluded=True,
    compatibility='Packed pair keeps FT/P16 but does not restore full-AA T1/T4. Baseline GN remains; GN-reuse override is explicitly absent. READ7 alters only admitted mm routes; scan retains dtype checks.',
    no_hidden_value_cache=True,same_arena_and_existing_resource_caps=True,
    setup_counts=dict(bound=687,lvfuse=649,psdb_fused=9,pair_block_indices=1,read7_fused=4),
    parity_layers=[0],all16_finite_required=True,original_S34_paired_controls=True,
    original_approximate_score_gates=True,no_additive_gain_claim=True,broad_qualified=False,serving_qualified=False,submitted=False,model_executed=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),
    protocol=pin(D/'protocol.json'),source_diff=pin(D/'source_diff.patch'),
    packed_source=pin(PP/'candidate.py'),packed_inverse=pin(PP/'source_inverse.json'),
    appended_fragments=[pin(R7/'fragment.py'),pin(SCAN/'fragment.py')],source_inverse_exact=True,
    changed_parent_functions=['_nonlinear_pair_batch','_predict_binary'],
    new_overrides=['_pb_bind via chained PP and READ7','_mm via READ7','_shared_old384_finite via scan'],
    gn_reuse_absent=True,nystrom_absent=True,model_executed=False))
original=read(T);c=rewrite(copy.deepcopy(original))
c['arms']['candidate']['source']=pin(D/'candidate.py')
c['arms']['candidate']['setup']['counts']=protocol['setup_counts'];c['parity_layers']=[0]
c['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'build.json',D/'protocol.json',D/'source_diff.patch',PP/'candidate.py',PP/'source_inverse.json',PP/'retained_component.json',R7/'fragment.py',R7/'retained_component.json',SCAN/'fragment.py',SCAN/'native2_assessment.json',H/'official_bank_S34_v1.json')]
c['interpretation']='Fixed packed-pair+READ7+scan combination. PP/READ7 admitted N8/runtime; scan isolated N2. GN-reuse and Nyström excluded. Same S34 controls and original N2/8 gates; no additive-gain or broader-qualification claim.'
assert c['arms']['base']==original['arms']['base'] and c['arms']['parity_parent']==original['arms']['parity_parent']
assert c['prospective_scope']=='native2_only'
assert all(c['stages'][stage][key]==original['stages'][stage][key] for stage in ('2','8') for key in ('gates','assessment_limits'))
once(D/'native_screen.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),build=pin(D/'build.json'),protocol=pin(D/'protocol.json'))))
