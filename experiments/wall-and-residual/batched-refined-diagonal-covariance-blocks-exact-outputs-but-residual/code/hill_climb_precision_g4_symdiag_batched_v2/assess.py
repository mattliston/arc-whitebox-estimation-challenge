"""Root-only frozen batched-v2 native2/8 assessment; saved outputs, no models/targets.

The original09dd failures remain false. Native byte parity is all16 layers;
whole-candidate economics versus S32 govern progression, not parent increment.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import os
import sys
import numpy as np

D = Path(__file__).resolve().parent
H2 = D.parent
NATIVE = H2.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
DRIVER = H2/'hill_climb_uvp_base_bcache_probe_v1/native_probe.py'
PARENT_STUDY = H2/'hill_climb_precision_g4_symdiag_v1'
CANDIDATE_SHA = 'e82f7bf1fbe736d37816492e42c456198df07006152436a299b36eee757e2341'
PARENT_SHA = '09dd5166b1edae161ef0b877d66df99b1891689ed2465280cf646d45dadacbfd'
S32_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
BUILD_SHA = 'ed53ec46a59caedea66601f40ef39ccc9a01fc6f3dc913bfb0f53860d90b29d9'
NATIVE_PROTOCOL_SHA = 'ccdb1ed5219a087a1b0da24cb971c4316760ec660e7b4d176dbd553aa782f256'
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
EVIDENCE = {}

def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

def checked(record):
    data = Path(record['path']).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
    EVIDENCE[record['path']] = record
    return data

def load(path):
    return json.loads(checked(pin(path)))

def guard(path, seconds, command=None):
    value = load(path)
    assert value['completed'] and value['exit_code'] == 0 and value['error'] is None
    assert value['whole_group_retired'] and value['remaining_rows'] == []
    assert value['rss_trigger_bytes'] == 7_500_000_000
    assert 0 <= value['max_rss_bytes'] < 7_500_000_000 and value['max_swap_growth_bytes'] == 0
    assert 0 <= value['elapsed_s'] <= seconds
    if command is not None:
        assert value['command'] == command
    return value

def metrics(base, current, ranks):
    b, c = ({row['rank']: row for row in rows} for rows in (base, current))
    assert set(b) == set(c) == set(ranks)
    delta = lambda key: 100*(sum(c[r][key] for r in ranks)/sum(b[r][key] for r in ranks)-1)
    bb = np.array([b[r]['adjusted'] for r in ranks])
    cc = np.array([c[r]['adjusted'] for r in ranks])
    idx = np.random.default_rng(20260926).integers(0, len(ranks), size=(50000, len(ranks)))
    ci = (100*np.quantile(cc[idx].mean(1)/bb[idx].mean(1)-1, [.025, .975])).tolist()
    return dict(adjusted_pct=delta('adjusted'), raw_pct=delta('mse'), bill_pct=delta('flops'),
                adjusted_wins=sum(c[r]['adjusted'] < b[r]['adjusted'] for r in ranks), adjusted_ci95_pct=ci,
                rows=[dict(rank=r, adjusted_pct=100*(c[r]['adjusted']/b[r]['adjusted']-1),
                           raw_pct=100*(c[r]['mse']/b[r]['mse']-1),
                           bill_pct=100*(c[r]['flops']/b[r]['flops']-1)) for r in ranks])


def prerequisites():
    assert pin(D/'native_protocol.json')['sha256']==NATIVE_PROTOCOL_SHA
    p=load(D/'native_protocol.json')
    run_pin=pin(D/'native_run_protocol.json')
    assert run_pin['sha256']=='fe5b3894d1341a2cfa4e8fab52b64458f491b8e1db0176f642348de1ade11089'
    run=json.loads(checked(run_pin))
    assert run['root_admitted'] is True and run['source_frozen'] is True and run['broad_admitted'] is False
    for key in ('candidate','build','primitive','primitive_guard','probe','driver'):
        assert run[key]==p[key]
    assert run['protocol']==p['source_protocol']
    assert run['resource_gates']==dict(rss_bytes=7500000000,added_swap_bytes=0,blas_threads=1,serial_root=True)
    for size in (2,6):
        st=p['stages'][str(size)]
        assert run[f'native{size}']==dict(ranks=st['ranks'],tag=st['tag'],guard_seconds=st['native_seconds'])
    assert run['native2_gates']==dict(adjusted_pct_at_most=-1,wins_min=2,all16_parent_bytes_exact=True)
    assert run['native8_gates']==dict(adjusted_pct_at_most=-1,wins_min=7,ci95_upper_below=0,all16_parent_bytes_exact=True)
    for key in ('candidate','build','source_protocol','policy','parent','bank_comparator','driver',
                'primitive','primitive_guard','probe','parent_native8','parent_native8_guard','template'):
        checked(p[key])
    for record in p['original_failure_receipts']:
        assert json.loads(checked(record))['passed'] is False
    assert p['candidate']==pin(D/'candidate.py') and p['candidate']['sha256']==CANDIDATE_SHA
    assert p['parent']==pin(PARENT_STUDY/'candidate.py') and p['parent']['sha256']==PARENT_SHA
    assert p['bank_comparator']['sha256']==S32_SHA
    assert p['build']['sha256']==BUILD_SHA and p['driver']==pin(DRIVER)
    b=json.loads(checked(p['build']))
    for key in ('parent','candidate','builder','protocol','fragment','wrapper','pricing','side_dependency','parent_build'):
        checked(b[key])
    assert b['candidate']==p['candidate'] and b['parent']==p['parent']
    assert b['protocol']==p['source_protocol'] and b['append_only'] is True and b['source_inverse_exact'] is True
    suffix=b'\n'+checked(b['fragment'])
    assert len(suffix)==b['suffix_bytes']==8656
    assert checked(b['candidate'])==checked(b['parent'])+suffix
    primitive=json.loads(checked(p['primitive']))
    assert primitive['passed'] and primitive['all_parent_bytes_exact'] and primitive['side_pool_reset_passed']
    assert primitive['source']==p['candidate'] and primitive['parent']==p['parent']
    assert primitive['build']==p['build'] and primitive['probe']==p['probe']
    assert primitive['protocol']==p['source_protocol'] and primitive['fragment']==b['fragment']
    assert primitive['pricing']==b['pricing'] and 0<=primitive['setup_s']<=4.1
    assert [r[1] for r in primitive['pb_log'] if r[0]=='bound']==[681]
    assert [r[1] for r in primitive['pb_log'] if r[0]=='lvfuse']==[649]
    assert [r[1] for r in primitive['pb_log'] if r[0]=='psdb_fused']==[8]
    assert [r['kind'] for r in primitive['cases']]==['integer_gram','gaussian_covariance']
    for case in primitive['cases']:
        assert [r['label'] for r in case['runs']]==['parent_first','child_first','parent_repeat','child_repeat']
        assert len({r['output_sha256'] for r in case['runs']})==1
        for r in case['runs']:
            assert r['parent_bytes_exact'] and r['flops']==(709895628 if r['label'].startswith('child') else 707798476)
            assert r['integer_reference_exact'] if case['kind']=='integer_gram' else r['relative_frobenius_error']<=2e-4
    guard(Path(p['primitive_guard']['path']),45,
          [str(NATIVE),'-B',p['probe']['path'],p['primitive']['path']])
    a=load(D/'native_admission.json')
    assert a['root_admitted'] is True and a['source_changes'] is False
    for key in ('candidate','build','policy','primitive','primitive_guard','probe','driver'):
        assert a[key]==p[key];checked(a[key])
    assert a['protocol']==p['source_protocol']
    assert a['native_protocol']==pin(D/'native_protocol.json')
    assert a['run_protocol']==run_pin
    assert a['assessor']==pin(__file__)
    checked(a['protocol']);checked(a['native_protocol']);checked(a['assessor'])
    original=json.loads(checked(p['parent_native8']))
    assert original['passed_integrity'] and original['admitted'] is False and original['n']==8
    assert original['candidate']==p['parent']
    assert original['gates']==dict(adjusted_gain=True,wins=True,ci_upper_negative=False)
    guard(Path(p['parent_native8_guard']['path']),120,
          [str(NATIVE),'-B',str(PARENT_STUDY/'assess.py'),'8'])
    return p,{r['path']:r for r in original['evidence']}


def panel(size,role,p,parent_evidence):
    stage=p['stages'][str(size)];ranks=stage['ranks']
    if role=='candidate':
        tag=stage['tag'];gp=D/stage['native_guard'];expected=CANDIDATE_SHA
    elif role=='parent':
        tag=f'hill_climb_precision_g4_symdiag_train{size}_v1'
        gp=PARENT_STUDY/f'native{size}_guard.json';expected=PARENT_SHA
    else:
        assert role=='s32'
        tag=f'hill_climb_star_projection32_train{size}_v1'
        gp=H2/'hill_climb_star_projection32_v1'/f'train{size}_guard.json';expected=S32_SHA
    folder=H2/tag
    if role!='candidate':
        for path in (folder/'manifest.json',folder/'rows.jsonl',gp):
            assert pin(path)==parent_evidence[str(path)]
    m=load(folder/'manifest.json')
    assert m['ranks']==ranks and m['runtime']==str(NATIVE) and m['numpy']=='2.4.6'
    assert m['thread_env']==dict.fromkeys(THREADS,'1')
    source=pin(m['candidate']);checked(source)
    assert source['sha256']==m['sha256']==expected
    if role=='candidate':assert source==p['candidate']
    if role=='parent':assert source==p['parent']
    command=None if role=='s32' else [str(NATIVE),'-B',str(DRIVER),source['path'],','.join(map(str,ranks)),tag]
    g=guard(gp,stage['native_seconds'],command)
    setup=m['setup']
    assert math.isfinite(setup['setup_s']) and 0<=setup['setup_s']<=4.1
    bound,fused={'candidate':(681,649),'parent':(673,649),'s32':(624,600)}[role]
    assert [r[1] for r in setup['pb_log'] if r[0]=='bound']==[bound]
    assert [r[1] for r in setup['pb_log'] if r[0]=='lvfuse']==[fused]
    if role=='candidate':assert [r[1] for r in setup['pb_log'] if r[0]=='psdb_fused']==[8]
    rows=[json.loads(line) for line in checked(pin(folder/'rows.jsonl')).decode().splitlines()]
    assert [r['rank'] for r in rows]==ranks
    predictions={}
    for r in rows:
        assert all(math.isfinite(r[k]) and r[k]>0 for k in ('mse','adjusted','util','wall_s','cpu_s'))
        assert type(r['flops']) is int and 0<r['flops']<=2**41
        assert r['util']==r['flops']/2**41
        assert math.isclose(r['adjusted'],r['mse']*max(.1,r['util']),rel_tol=1e-14,abs_tol=0)
        record=pin(folder/f'pred_{r["rank"]}.npy')
        if role!='candidate':assert record==parent_evidence[record['path']]
        checked(record);predictions[r['rank']]=record
        if role!='s32':
            u=r['ubc'];assert u['retired'] is True
            assert u['retained_cache_bytes']==u['live_cache_bytes']==0
            assert 0<=u['max_resident_bytes']<=512000000
    return rows,predictions,g['max_rss_bytes'],dict(role=role,tag=tag,**setup)


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ('2','8')
    assert Path(sys.executable).absolute()==NATIVE and np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in THREADS)
    count=int(sys.argv[1]);target=D/f'native{count}_assessment.json'
    assert not target.exists()
    p,parent_evidence=prerequisites()
    if count==8:
        prior=load(D/'native2_assessment.json')
        assert prior['passed_integrity'] and prior['admitted'] and prior['n']==2
        assert prior['candidate']==p['candidate'] and prior['native_protocol']==pin(D/'native_protocol.json')
        assert prior['assessor']==pin(__file__) and prior['all16_parent_bytes_exact']
        for record in prior['evidence']:checked(record)
        guard(D/'native2_assess_guard.json',120,[str(NATIVE),'-B',str(D/'assess.py'),'2'])
    rows={role:[] for role in ('s32','parent','candidate')}
    predictions={role:{} for role in rows};peaks=[];setups=[]
    for size in ((2,) if count==2 else (2,6)):
        for role in rows:
            rr,pp,peak,setup=panel(size,role,p,parent_evidence)
            rows[role].extend(rr);predictions[role].update(pp);peaks.append(peak);setups.append(setup)
    ranks=sorted(r['rank'] for r in rows['candidate'])
    assert len(set(ranks))==count
    for rank in ranks:
        arrays={}
        for role in rows:
            array=np.load(io.BytesIO(checked(predictions[role][rank])),allow_pickle=False)
            assert array.shape==(16,1024) and array.dtype==np.dtype('float64') and np.isfinite(array).all()
            arrays[role]=array
        assert arrays['candidate'].tobytes()==arrays['parent'].tobytes(),('all16',rank)
        del arrays,array
    parent=metrics(rows['parent'],rows['candidate'],ranks)
    s32=metrics(rows['s32'],rows['candidate'],ranks)
    assert parent['raw_pct']==0
    expected=p['stages'][str(count)]['gates']
    assert expected==dict(adjusted_pct_at_most=-1,wins_min=2 if count==2 else 7,ci95_upper_below=0)
    gates=dict(adjusted_gain=s32['adjusted_pct']<=-1,
               wins=s32['adjusted_wins']>=expected['wins_min'],ci_upper_negative=s32['adjusted_ci95_pct'][1]<0)
    result=dict(passed_integrity=True,admitted=all(gates.values()),n=count,ranks=ranks,**s32,
        candidate=p['candidate'],protocol=p['source_protocol'],native_protocol=pin(D/'native_protocol.json'),
        native_admission=pin(D/'native_admission.json'),assessor=pin(__file__),policy=p['policy'],gates=gates,
        run_protocol=pin(D/'native_run_protocol.json'),
        component_comparator=p['parent'],bank_comparator=p['bank_comparator'],versus_s32=s32,versus_parent=parent,
        bootstrap=p['bootstrap'],maximum_guard_rss_bytes=max(peaks),setup_records=setups,
        evidence=list(EVIDENCE.values()),all16_parent_bytes_exact=True,all16_predictions_finite=True,
        original_v1_failures_preserved=True,original_v1_results=p['original_failure_receipts'],
        parent_original_training_admitted=False,parent_result=p['parent_native8'],primitive=p['primitive'],
        raw_error_veto=False,bank_progression_only=True,bank_admitted=False,
        targets_read=False,model_executed=False,serving_qualified=False,submitted=False)
    for record in list(EVIDENCE.values()):checked(record)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('admitted','n','adjusted_pct','raw_pct','bill_pct','adjusted_wins','adjusted_ci95_pct','gates','versus_parent')}))


if __name__=='__main__':main()
