from pathlib import Path
import json,hashlib,types,sys
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_protected_old_projection_v1';Z=O/'lead/runs/astra_rtt2_prefix_z1_late_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();C=D/'candidate_protected_old_QR_ON_HELD_v1.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub;m=types.ModuleType('_pop_import');m.__file__=str(C);sys.modules[m.__name__]=m;exec(compile(C.read_text(),str(C),'exec'),m.__dict__)
assert m._RC_PARENT_READ is m._rtt_read and m._RTT_PARENT_INNER is m._pop_inner and m.Estimator.predict is m._pop_predict
assert m._RTT_ACTIVE is False and m._RTT_CONTEXT==[None] and m._RTP_ON
R=D/'SOURCE_IMPORT_ONLY_v1.json';R.write_text(json.dumps(dict(candidate_sha256=sha(C),full_import_pass=True,inner_capture_binding_pass=True,read_binding_unchanged=True,no_setup_predict_models=True),indent=2)+'\n')
f=D/'finite_pop_astra_v1.py';s=(Z/'finite_pzl_astra_v1.py').read_text().replace('pzl','pop').replace('PZL','POP').replace('[:10]','[:7]');s=s.replace("'_RTT_CONTEXT')","'_RTT_CONTEXT','_POP_STATS')")
s=s.replace("            evaluations.append(dict(role='ON',seed=seed,**item))",'''            pop=st['_POP_STATS'];assert pop['retired'] is True
            assert [x['layer'] for x in pop['reads']]==list(range(7,15))
            for rec in pop['reads']:
                assert rec['center']==((3*rec['m']+255)//256)*64-2
                if rec['changed']:assert rec['basis_rank']==2 and 0<rec['center']<rec['center']+2<rec['m']
            evaluations.append(dict(role='ON',seed=seed,**item))''')
f.write_text(s);j=D/'native_pop_judge_astra_v1.py';j.write_text((Z/'native_pzl_judge_astra_v1.py').read_text().replace('pzl','pop'))
g=D/'grade_retired_phase_v1.py';g.write_text((Z/'grade_retired_phase_v1.py').read_text().replace('pzl','pop').replace('compute_hunt','bigger_gain_hunt').replace('[:10]','[:7]'))
c=json.loads((Z/'PROTOCOL_REGISTERED_astra_v1.json').read_text());c['purpose']='pop_qr_e406_pilot8_fresh16_oos8';c['candidates']['ON'].update(path=str(C),sha256=sha(C));c['source_pins']={k:v for k,v in c['source_pins'].items() if not k.startswith(str(Z))}
files=[f,j,g,R,D/'SOURCE_READY_HELD_v1.json',D/'fragment_protected_old_HELD_v1.py',D/'build_source_and_tiny_HELD_v1.py',O/'lead/runs/astra_read_column_oracle_v1/PROTECTED_OUTER_QUARTER_TARIFF_v1.json',O/'lead/runs/astra_read_column_oracle_v1/TRAIN16_OLD_PROTECTED_QUARTER_v1.json'];c['source_pins'].update({str(p):sha(p) for p in files})
c.update(executed_by='bigger_gain_hunt',approved=True,reviewed_by='root',approved_by='root',conditional_root_authorization='Root message: 1.470% forecast merits one real pilot despite proxy risk. Freeze EXACT central75%+two vc/mu projected rows rule already priced; young unchanged/diagonal exact. Build source/import/tiny then own pilot8 ON-only vs cachedQR+e406 using late-z1/NP fast harness. Four native slots authorized after compute MLP retires. Negative both pilot -> remaining16 direct; finalall24+OOS8 negative both. No fraction/basis tuning. Follow-up root: MLP actualretirement84fad0f6 complete; protected-old pilot may launch four slots after source ready.',run_names={p:{str(i):f'run_pop_{p}_s{i}_v1' for i in range(4)} for p in ('pilot','remaining')},source_readiness=dict(path=str(D/'SOURCE_READY_HELD_v1.json'),sha256=sha(D/'SOURCE_READY_HELD_v1.json')),readiness=json.loads((D/'SOURCE_READY_HELD_v1.json').read_text()),root_source_decision='Root accepted1.470% warm prior with approximation risk; fixed protected old-only arm.')
c['rule']['source']='L7..14 fixed center=ceil(.75*m/64)*64-2 plus two vc/mu projected rows in one contraction; fullm inner restored; all young arithmetic and scalar diagonal untouched; no tuning'
c['qualification_protocol_path']=str(R);c['qualification_protocol_sha256']=sha(R);c['qualification_root_retirement_path']=str(R);c['qualification_root_retirement_sha256']=sha(R)
p=D/'PROTOCOL_REGISTERED_astra_v1.json';p.write_text(json.dumps(c,indent=2)+'\n')
for t in (f,j,g):compile(t.read_text(),str(t),'exec')
print(json.dumps(dict(candidate=sha(C),protocol=sha(p),runner=sha(f),judge=sha(j),import_receipt=sha(R))))
