import ast,copy,hashlib,json,pathlib,subprocess
O=pathlib.Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'scout_card/capacity_recon_prefix_held_astra_v1';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
parent=O/'lead/runs/astra_energy_e1_standalone_v1/finite_e1_science_UNVALIDATED_v2.py';helper=D/'finite_recon_smoke_UNVALIDATED_astra_v2.py'
s=parent.read_text().replace('O=D.parents[2]','O=D.parents[1]').replace("H=D/'finite_e1_smoke_UNVALIDATED_v3.py'","H=D/'finite_recon_smoke_UNVALIDATED_astra_v2.py'").replace('afa823f8b2122cd46b79f7e8e0a13a44ffebeaeca240676d0230e44da686ce1e',sha(helper)).replace("MODES={'C':0,'E1':1}","MODES={'C':'control','RP':'recon_prefix'}").replace("'E1'","'RP'").replace("'energy_e1_", "'recon_prefix_").replace('energy_e1_standalone_fresh16','recon_prefix_fresh16').replace("'e1s_", "'rps_").replace("f'e1s_", "f'rps_").replace('Owned standalone E1','Owned reconstruction-prefix')
s=s.replace('Fresh evaluation may overlap price. The fixed MAX price must be frozen before\nany fresh mean.','The fixed MAX price must be frozen before preparation or launch.\nAll scientific means require complete fixed pools and explicit root reaping.')
s=s.replace("assert c['adapter_path']==base['adapter_path']", "assert c['runtime_path']==base['runtime_path'] and c['private_torch_path']==base['private_torch_path']")
s=s.replace("'ROOT_CONTROLLER_RETIRED_v1.json').read_text())\n    assert root['approved']", "'ROOT_PHASE_RETIREMENT_v1.json').read_text())\n    assert root['approved']",1).replace("root['managed_session_exit_code']==0\n    assert root['phase_owner_sha256']", "root['managed_exit_code']==0\n    assert root['owner_sha256']",1)
s=s.replace("    return c\ndef run_dir", "    assert c['fresh']==list(range(7102440,7102456))\n    return c\ndef run_dir")
def swap(name,body):
 global s
 node=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==name);lines=s.splitlines(True);s=''.join(lines[:node.lineno-1])+body.strip()+'\n'+''.join(lines[node.end_lineno:])
swap('price', '''def price(c):
    p=Path(c['native_price_path']);assert sha(p)==c['native_price_sha256']
    r=json.loads(p.read_text());assert r['approved'] is True and r['all8_clean_retired'] is True
    assert r['protocol_sha256']==c['native_price_protocol_sha256']
    qp=Path(r['qualification_path']);assert sha(qp)==r['qualification_sha256']
    q=json.loads(qp.read_text());assert q['status']=='native8_complete_all_guards_retired'
    assert q['protocol_sha256']==r['protocol_sha256'] and q['all_guards_retired'] is True and q['owned_group_retirement_verified'] is True
    owner=Path(r['owner_retirement_path']);assert sha(owner)==r['owner_retirement_sha256']
    root=json.loads(owner.read_text());assert root['managed_exit_code']==0 and root['controller_absent'] is True and root['all8_clean_retired'] is True
    assert root['qualification_sha256']==sha(qp)
    order=[(role,seed) for seed in (7100136,7100137) for role in ('current320','full384','OFF','ON')]
    assert [(e['role'],e['seed']) for e in q['evaluations']]==order
    assert [(e['role'],e['seed']) for e in q['checks']]==order and all(all(e['check'].values()) for e in q['checks'])
    bills={}
    for e in q['evaluations']:
        cp=Path(e['certificate']);gp=Path(e['guard']);pp=cp.with_name(cp.stem+'_pred.npz')
        assert sha(cp)==e['certificate_sha256'] and sha(gp)==e['guard_sha256'] and sha(pp)==e['prediction_npz_sha256']
        g=json.loads(gp.read_text());assert g['completed'] is True and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] is True and g['remaining_rows']==[]
        assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
        cert=json.loads(cp.read_text());assert cert['sha256']==c['native_candidates'][e['role']]['sha256']
        assert len(cert['rows'])==1 and cert['rows'][0]['seed']==e['seed']
        cost=cert['rows'][0]['flops'];assert type(cost) is int and cost>0
        bills[e['role'],e['seed']]=cost
    logs=[math.log(bills['ON',seed]/bills['current320',seed]) for seed in (7100136,7100137)]
    assert q['price']==r['price'] and q['price']['aggregation']=='max_pair_log_ratio'
    assert math.isclose(max(logs),q['price']['actual_total_log_factor'],rel_tol=1e-14,abs_tol=1e-16)
    assert q['price']['cost_upside_both_seeds'] is True and all(bills['ON',seed]<bills['full384',seed] for seed in (7100136,7100137))
    return {'RP':max(logs)},sha(p)''')
swap('expected_request', '''def expected_request(c,phase,item):
    assert item['arm'] in MODES and item['mode']==MODES[item['arm']] and item['seed'] in c[phase]
    r=json.loads(Path(c['template_request']).read_text())
    folder=Path(c['private_torch_path'])/('cutsim_'+item['mode'])
    names=('torch_cutsim_s46_v11.py','n6y_cutsim_v2.py','torch_rowcuts_v15.py','torch_push_native_v2.py','pbase_cutsim_v1.py')
    sources={n:c['source_pins'][str(folder/n)] for n in names}
    r.update(id=item['id'],rank=item['seed'],keep_final=True,keep_all=True)
    r['cutsim']=dict(module=str(folder/names[0]),sha256=sources[names[0]],sources=sources,opts=r['cutsim']['opts'])
    return r''')
s=s.replace("    assert type(r['astra_energy_e1']) is int and r['astra_energy_e1']==item['mode']\n    a=r['astra_e1_provenance']", "    a=r['astra_recon_provenance']")
s=s.replace("a['adapter_sha256']==c['source_pins'][c['adapter_path']]", "a['runtime_path']==c['runtime_path'] and a['runtime_sha256']==c['source_pins'][c['runtime_path']]")
s=s.replace("    # No price() call: fresh evaluations may overlap native price after smokes.", "    price(c)  # Fixed native8 bill and all8 retirement precede any scientific launch.")
s=s.replace("    if a.action=='prepare':\n        assert a.grant", "    if a.action=='prepare':\n        price(c)\n        assert a.grant")
fn=D/'finite_recon_science_UNVALIDATED_astra_v1.py';compile(s,str(fn),'exec');fn.open('x').write(s)
base=json.loads((D/'SMOKE_PROTOCOL_HELD_astra_v2.json').read_text());c={k:copy.deepcopy(base[k]) for k in ('source_pins','template_request','worker_path','runtime_path','private_torch_path','guard_python','model_python','fresh_allocation_path','fresh_allocation_sha256','native_price_path','native_price_sha256')}
c.update(status='HELD_DRAFT_NO_REQUESTS_OR_EVALUATIONS',approved=False,reviewed_by='PENDING_ROOT',base_protocol=str(D/'SMOKE_PROTOCOL_astra_v1.json'),base_protocol_sha256='PENDING_ROOT',fresh=list(range(7102440,7102456)),oos=list(range(7100592,7100608)),runs={'fresh':'run_recon_fresh_astra_v1','oos':'run_recon_oos_astra_v1'},native_price_protocol_sha256=sha(D/'NATIVE_PRICE_PROTOCOL_astra_v1.json'),native_candidates=json.loads((D/'NATIVE_PRICE_PROTOCOL_astra_v1.json').read_text())['candidates'],physics='One fixed full384-state/fullread approximation with prefix reconstruction v320 only L10-13. No rank/layer/refit search.',fresh_gate='Complete16 strict negative actualnet; MAX native136137 frozen before any request or mean.',oos_gate='New ownC+RP16 on fixed592-607; complete fresh16/OOS16/all32 strictnegative actualnet.',runtime='OriginalTorch2/sixBLAS2; outerall7BLAS1/controllernice0/guardnice10once;7.5GBcaps/free50/count10, explicitroot1..3slots; allrootreaped beforemeans.',old_failures='Oldjoint/fund64/earlyreadFAIL preserved. No reuse of their scientific decisions.',scope='Source-only held. Separate root approved registered source, allocation, costfreeze, complete smoke3, source-specific reuse, explicitphase grants required. No new requests.',native_followon='Separate exactCzero584+C/ON2440/2441 formalv5; native10fixed136-145 latestparent negative; owncensus/prebind/seal/release/SAFE/prefire; no implicitgrant.')
for p in (fn,parent,D/'NATIVE_PRICE_PROTOCOL_astra_v1.json'):c['source_pins'][str(p)]=sha(p)
proto=D/'FOLLOWUP_PROTOCOL_HELD_astra_v1.json';proto.open('x').write(json.dumps(c,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(helper=sha(fn),protocol=sha(proto),pins=len(c['source_pins']),no_requests=True)))
