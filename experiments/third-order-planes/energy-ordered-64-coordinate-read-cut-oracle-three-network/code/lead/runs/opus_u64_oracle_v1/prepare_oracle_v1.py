"""ORACLE (accuracy-only, cost ignored): U64 candidate with energy-ordered 64-direction drop at each of the 7 cut events. 3 seeds vs exact 2b108 controls. No model calls."""
from pathlib import Path
import hashlib,json,datetime
D=Path(__file__).resolve().parent;R=D.parent;S=R/'opus_u64_screen_v1';V=R/'astra_read256_inner_cost_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    p=D/name
    with p.open('x') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,indent=2);f.write('\n')
    return p
SEEDS=[7103249,7103252,7103254]
src=(V/'candidate_uniform64_read_HELD_v1.py').read_text();assert sha(V/'candidate_uniform64_read_HELD_v1.py').startswith('24b13a67')
old='''        keep = rank - 64
        reduced = _ubx_product(Dv, views(views(Fv2[:keep, :]).T), uv, 'vt')
        inner = fnp.empty((rows, rank), dtype=fnp.float32)
        top, tail = views(fnp.split(inner, [keep], axis=1))
        h_(fnp.copyto(top, reduced))
        h_(fnp.copyto(tail, 0))
        del top, tail, reduced
        views.clear()
'''
new='''        keep = rank - 64
        # ORACLE ONLY (unmetered numpy): drop the 64 lowest-energy right-singular directions of the full vt product.
        import numpy as _onp
        full = _ubx_product(Dv, Fv2.T, uv, 'vt')
        _x = _onp.array(full, copy=True).astype(_onp.float64)
        _s, _vt = _onp.linalg.svd(_x, full_matrices=False)[1:]
        _vk = _vt[:keep]
        inner = fnp.asarray(((_x @ _vk.T) @ _vk).astype(_onp.float32))
        _tot = float((_s ** 2).sum())
        _ASTRA_READ64_STATS.setdefault('oracle', []).append((layer, rank, float((_s[keep:] ** 2).sum()) / _tot, float((_x[:, keep:] ** 2).sum()) / _tot))
        del full, _x, _s, _vt, _vk
'''
assert src.count(old)==1
write('candidate_u64_oracle_svd_NOTDEPLOYABLE_v1.py',src.replace(old,new))
j=(S/'native_u64_screen_judge_v1.py').read_text()
for a,b in [("science_protocol['purpose']=='opus_u64_screen_v1'","science_protocol['purpose']=='opus_u64_oracle_v1'"),
            ("assert set(phase_seeds)=={'screen'} and len(phase_seeds['screen'])==9","assert set(phase_seeds)=={'screen'} and len(phase_seeds['screen'])==3"),
            ("allowed=phase_seeds['screen'];assert allowed==list(range(7103249,7103258))","allowed=phase_seeds['screen'];assert allowed==[7103249,7103252,7103254]")]:
    assert j.count(a)==1,a;j=j.replace(a,b)
write('native_u64_oracle_judge_v1.py',j)
write('u64_screen_contract_v1.py',(S/'u64_screen_contract_v1.py').read_text())
r=(S/'finite_u64_screen_v2.py').read_text()
for a,b in [("q.guard.JUDGE=D/'native_u64_screen_judge_v1.py'","q.guard.JUDGE=D/'native_u64_oracle_judge_v1.py'"),
            ("p=D/'PROTOCOL_REGISTERED_opus_v2.json'","p=D/'PROTOCOL_REGISTERED_oracle_v1.json'"),
            ("c['purpose']=='opus_u64_screen_v1'","c['purpose']=='opus_u64_oracle_v1'"),
            ("assert grant['maximum_native_calls']==9 and","assert grant['maximum_native_calls']==3 and"),
            ("h=D/f'run_u64screen_{phase}_s{shard}_v1'","h=D/f'run_oracle_{phase}_s{shard}_v1'")]:
    assert r.count(a)==1,a;r=r.replace(a,b)
write('finite_u64_oracle_v1.py',r)
cand=D/'candidate_u64_oracle_svd_NOTDEPLOYABLE_v1.py'
write('OPUS_LEAD_GRANT_v1.json',dict(approved=True,reviewed_by='root',root='opus lead directive 2026-10-16 oracle follow-up (<=45 min, <=2 natives)',source_sha256=sha(cand),seeds=SEEDS,maximum_native_calls=3,max_concurrent_native=2,no_fire_authority=True,not_deployable=True))
c=json.loads((S/'PROTOCOL_REGISTERED_opus_v2.json').read_text())
ctl=[next(e for e in c['stack_control_reuse'] if e['seed']==s) for s in SEEDS]
c.update(purpose='opus_u64_oracle_v1',phase_seeds={'screen':SEEDS},identity_groups={'screen':SEEDS},native_calls_max=3,stack_control_reuse=ctl,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         rule=dict(scope='ORACLE accuracy-only: energy-ordered (SVD of full vt product) 64-direction drop at the same 7 L10-14 events; cost ignored; not deployable.',decision='Pre-registered: mean MSE log change on 7103249/52/54 <= +0.15% passes; otherwise close read-truncation family.'),run_names=['run_oracle_screen_s0_v1','run_oracle_screen_s1_v1'],decision_names=['ORACLE3_RESULT_v1.json'])
c['candidates']['ON']=dict(path=str(cand),sha256=sha(cand),mode=1)
for n in ('finite_u64_screen_v1.py','finite_u64_screen_v2.py','native_u64_screen_judge_v1.py','u64_screen_contract_v1.py','OPUS_LEAD_GRANT_v1.json','prepare_u64_screen_v1.py','prepare_u64_screen_v2.py','PROTOCOL_REGISTERED_opus_v1.json'):c['source_pins'].pop(str(S/n),None)
for p in (cand,D/'finite_u64_oracle_v1.py',D/'native_u64_oracle_judge_v1.py',D/'u64_screen_contract_v1.py',D/'OPUS_LEAD_GRANT_v1.json',Path(__file__).resolve()):c['source_pins'][str(p)]=sha(p)
for path,digest in c['source_pins'].items():assert sha(path)==digest,path
write('PROTOCOL_REGISTERED_oracle_v1.json',c)
for n in ('finite_u64_oracle_v1.py','native_u64_oracle_judge_v1.py','u64_screen_contract_v1.py','candidate_u64_oracle_svd_NOTDEPLOYABLE_v1.py'):compile((D/n).read_text(),n,'exec')
print('oracle',sha(cand)[:12],'protocol',sha(D/'PROTOCOL_REGISTERED_oracle_v1.json')[:12])
