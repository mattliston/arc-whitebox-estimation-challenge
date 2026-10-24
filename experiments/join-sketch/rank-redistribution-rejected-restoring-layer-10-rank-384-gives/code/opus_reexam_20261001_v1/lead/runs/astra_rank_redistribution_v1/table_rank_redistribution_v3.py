"""HELD complete-pool rank grading; never exposes partial scientific summaries."""
import hashlib,json,math,statistics,sys,re,struct
from pathlib import Path
HERE=Path(__file__).resolve().parent;O=HERE.parents[2]
WEIGHTED=O/'lead/runs/astra_newideas_v1'
HARNESS=O/'scout_card/rank_redistribution_held_astra_v1'
ARMS={'fund64':'fund','transfer64':'joint'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prereg():
    d=json.loads((HERE/'PREREG_v1.json').read_text())
    assert d['status']=='root_registered_before_fresh_evaluations'
    assert d['joint_allocation_registered'] is True
    assert d['arms']=={'fund64':{'7':320,'10':320,'13':384},'transfer64':{'7':320,'10':384,'13':384}}
    assert d['r_u']==256 and d['law_fit_blend']==0
    assert sha(d['baseline_params'])==d['baseline_params_sha256']
    assert d['screen_seeds']==list(range(7101096,7101112))
    assert d['confirmation_seeds']==list(range(7100592,7100608))
    for key,name in [('harness_manifest_sha256',HARNESS/'harness_manifest_astra_v1.json'),
                     ('allocation_receipt_sha256',HERE/'JOINT_ALLOCATION_AUDIT_v1.json'),
                     ('oos_control_provenance_sha256',HERE/'OOS_CONTROL_PROVENANCE_v1.json')]:
        assert sha(name)==d[key],key
    manifest=json.loads((HARNESS/'harness_manifest_astra_v1.json').read_text())
    for name,digest in manifest['files'].items():assert sha(HARNESS/name)==digest,name
    for name,digest in manifest['parent_files'].items():assert sha(WEIGHTED/name)==digest,name
    for name,digest in manifest['shared_sources'].items():assert sha(O/name)==digest,name
    for name,digest in d['helpers'].items():assert sha(HERE/name)==digest,name
    npath=Path(d['native_price_manifest']);assert sha(npath)==d['native_price_manifest_sha256']
    native=json.loads(npath.read_text())
    assert sha(native['parent']['path'])==native['parent']['sha256']
    for a,spec in native['arms'].items():
        assert sha(spec['candidate'])==spec['candidate_sha256'],a
        assert sha(spec['fragment'])==spec['fragment_sha256'],a
    assert d['native_price_aggregation']=='max_pair_log_ratio'
    return d
def bits(x):
    if isinstance(x,list):return tuple(bits(v) for v in x)
    return struct.pack('>d',float(x))
def path(phase,arm,seed):return HERE/('run_'+phase)/'results'/f'rdist_{arm}_r{seed}.json'
def read(p):
    assert not p.with_suffix('.error').exists(),str(p.with_suffix('.error'))
    d=json.loads(p.read_text())
    assert d['id']==p.stem and d['rank']==int(re.search(r'_r(\d+)(?:_|$)',p.stem).group(1))
    assert len(d['mse_layers'])==16 and all(math.isfinite(x) and x>0 for x in d['mse_layers'])
    return d
def ref(seed,confirmation=False):
    return (O/'physicist/t141_k4l_ext/results'/f'ph141_r{seed}_c.json' if confirmation
            else WEIGHTED/'run_screen/results'/f'wfit_z_r{seed}.json')
def identity():
    prereg();p=path('smoke','zero',7100584)
    w=WEIGHTED/'run_smoke/results/wfit_z_r7100584.json'
    old=O/'physicist/t141_k4l_ext/results/ph141_r7100584_c.json'
    if not p.exists() or not w.exists():return dict(complete=False,passed=False)
    a,b,c=map(read,(p,w,old));keys=('mse','mse_layers','final_prediction')
    checks={k:bits(a[k])==bits(b[k])==bits(c[k]) for k in keys}
    assert len(a.get('prediction',[]))==16,'Private zero must capture all16 layers'
    return dict(complete=True,passed=all(checks.values()),checks=checks,
                result_sha256={str(q):sha(q) for q in (p,w,old)})
def effects():
    control=identity()
    files=[path('effects',a,7100584) for a in ARMS]
    if not control['complete'] or not all(p.exists() for p in files):return dict(complete=False,passed=False)
    assert control['passed'];zero=read(path('smoke','zero',7100584))
    rows={}
    for arm,p in zip(ARMS,files):
        d=read(p);pred=d.get('prediction',[]);assert len(pred)==16
        changed=[i for i,(x,y) in enumerate(zip(pred,zero['prediction'])) if bits(x)!=bits(y)]
        rows[arm]=dict(unchanged_0_to_6=bits(pred[:7])==bits(zero['prediction'][:7]),
                       first_changed_layer=min(changed) if changed else None,
                       final_changed=bits(d['final_prediction'])!=bits(zero['final_prediction']),result_sha256=sha(p))
    passed=all(r['unchanged_0_to_6'] and r['first_changed_layer'] is not None and
               r['first_changed_layer']>=7 and r['final_changed'] for r in rows.values())
    return dict(complete=True,passed=passed,arms=rows,smoke_gains_not_used=True)
def coverage(arms,confirmation=False):
    d=prereg();seeds=d['confirmation_seeds' if confirmation else 'screen_seeds'];phase='confirm' if confirmation else 'screen'
    errors=[str(path(phase,a,s).with_suffix('.error')) for a in arms for s in seeds if path(phase,a,s).with_suffix('.error').exists()]
    assert not errors,errors
    counts={a:sum(path(phase,a,s).exists() for s in seeds) for a in arms}
    refs=sum(ref(s,confirmation).exists() for s in seeds)
    return dict(complete=all(n==16 for n in counts.values()) and refs==16,counts=counts,reference_count=refs)
def prices():
    # Root freezes ONLY actual FLOP pairs before any fresh summary. The receipt
    # stores paths/hashes of complete native results, not selected MSE values.
    p=HERE/'NATIVE_PRICE_v1.json';d=json.loads(p.read_text())
    assert d['status']=='root_frozen_before_fresh_summary'
    assert d['prereg_sha256']==sha(HERE/'PREREG_v1.json')
    assert set(d['arms'])==set(ARMS)
    assert d['aggregation']=='max_pair_log_ratio'
    cfg=prereg()
    native=json.loads(Path(cfg['native_price_manifest']).read_text())
    expected={'base':native['parent']['sha256']}
    expected.update({a:native['arms'][a]['candidate_sha256'] for a in ARMS})
    assert d['candidate_source_sha256']==expected
    out={} 
    for arm,pairs in d['arms'].items():
        assert [x['seed'] for x in pairs]==[7100136,7100137]
        values=[]
        for x in pairs:
            bills={}
            for role in ('base','arm'):
                q=Path(x[role+'_path']);assert sha(q)==x[role+'_sha256']
                record=json.loads(q.read_text())
                assert record['sha256']==d['candidate_source_sha256'][role if role=='base' else arm]
                hits=[r for r in record['rows'] if r.get('seed',r.get('rank'))==x['seed']]
                assert len(hits)==1 and math.isfinite(hits[0]['flops']) and hits[0]['flops']>0
                bills[role]=hits[0]['flops']
            values.append(math.log(bills['arm']/bills['base']))
        out[arm]=max(values)  # Fixed conservative maximum; no arm/seed price selection.
    return out,sha(p)
def delta(arm,confirmation=False):
    assert coverage([arm],confirmation)['complete'],'No partial scientific data'
    d=prereg();seeds=d['confirmation_seeds' if confirmation else 'screen_seeds'];phase='confirm' if confirmation else 'screen'
    return [math.log(read(path(phase,arm,s))['mse_layers'][-1]/read(ref(s,confirmation))['mse_layers'][-1]) for s in seeds]
def stats(raw,bill):
    return dict(n=len(raw),mean_mse_log_pct=100*statistics.mean(raw),bill_log_pct=100*bill,bill_aggregation='max_pair_log_ratio',
                mean_net_log_pct=100*(statistics.mean(raw)+bill),
                se_pct=100*statistics.stdev(raw)/math.sqrt(len(raw)),lower_net=sum(x+bill<0 for x in raw))
def screen():
    c=coverage(ARMS)
    if not c['complete']:return dict(**c,survivors=[])
    assert effects()['passed'];costs,price_sha=prices()  # before reading fresh values
    rows={a:stats(delta(a),costs[a]) for a in ARMS}
    return dict(complete=True,arms=rows,survivors=[a for a in ARMS if rows[a]['mean_net_log_pct']<0],native_price_sha256=price_sha)
def confirm():
    decision=screen();assert decision['complete'] and decision['survivors']
    c=coverage(decision['survivors'],True)
    if not c['complete']:return dict(**c,survivors=decision['survivors'])
    costs,price_sha=prices();rows={}
    for a in decision['survivors']:
        fresh,oos=delta(a),delta(a,True);full=stats(fresh+oos,costs[a]);sep=stats(oos,costs[a])
        rows[a]=dict(all32=full,oos16=sep,passed=full['mean_net_log_pct']<0 and sep['mean_net_log_pct']<0)
    return dict(complete=True,arms=rows,passed=[a for a in rows if rows[a]['passed']],native_price_sha256=price_sha)
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('identity','effects','screen','confirm')
    print(json.dumps(globals()[sys.argv[1]](),indent=2,allow_nan=False))
