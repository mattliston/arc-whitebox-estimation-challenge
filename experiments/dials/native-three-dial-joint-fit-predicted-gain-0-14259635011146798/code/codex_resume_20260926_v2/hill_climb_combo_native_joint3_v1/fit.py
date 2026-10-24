"""Root-only one native three-dial box step from six saved fixed FIT4 arms."""
from pathlib import Path
import hashlib, itertools, json, math, os, sys
import numpy as np

D=Path(__file__).resolve().parent
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
EVIDENCE={}

def pin(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)

def verified(record):
    assert pin(record['path'])==record,record['path']; EVIDENCE[record['path']]=record
    return Path(record['path'])

def read_saved(path):
    p=verified(pin(path));return json.loads(p.read_text())

def once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def prediction(record):
    a=np.load(verified(record),allow_pickle=False)
    assert a.shape==(16,1024) and a.dtype==np.float64 and np.isfinite(a).all()
    assert np.array_equal(a,a.astype(np.float32).astype(np.float64))
    return a

def checked_guard(g,command,seconds,rss):
    assert g['command']==command and g['completed'] and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] and g['remaining_rows']==[] and g['max_swap_growth_bytes']==0
    assert g['rss_trigger_bytes']==rss and 0<g['max_rss_bytes']<rss and 0<g['elapsed_s']<=seconds and g['samples']>0

def box_solve(G,g):
    # Ridge makes every nonzero-column principal subproblem positive definite.
    active=[i for i in range(3) if G[i,i]>0]
    Q=G+.1*np.diag(np.diag(G)); best=None; feasible=0
    for state in itertools.product((-1,0,1),repeat=len(active)):
        x=np.zeros(3);free=[i for i,s in zip(active,state) if s==0];fixed=[i for i,s in zip(active,state) if s!=0]
        for i,s in zip(active,state):x[i]=s
        if free:x[free]=np.linalg.solve(Q[np.ix_(free,free)],-g[free]-Q[np.ix_(free,fixed)]@x[fixed])
        if not np.isfinite(x).all() or np.any(np.abs(x)>1+1e-12):continue
        x=np.clip(x,-1,1);value=float(x@Q@x+2*g@x);feasible+=1
        if best is None or value<best[0]:best=(value,x.copy())
    assert best is not None
    x=best[1];grad=Q@x+g;scale=max(float(np.max(np.abs(Q))),float(np.max(np.abs(g))),1e-300)
    for i in active:
        assert (grad[i]>=-1e-9*scale if x[i]<=-1 else grad[i]<=1e-9*scale if x[i]>=1 else abs(grad[i])<=1e-9*scale)
    return x,Q,feasible

def main():
    assert len(sys.argv)==2 and pin(D/'protocol.json')['sha256']==sys.argv[1]
    c=read_saved(D/'protocol.json'); assert c['root_admitted'] and c['fit_ranks']==[35,100,36,45]
    assert c['check_ranks']==[52,58,64,70,76,80,84,88] and c['one_endpoint_only']
    assert c['step_fraction']==.02 and c['ridge_diagonal_fraction']==.1 and c['box']==[-1.,1.]
    assert c['fit_source']==pin(__file__) and str(Path(sys.executable).absolute())==c['runtime'] and np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in THREADS)
    for key in ('root_selection','baseline','baseline_native8','roles','store_manifest','driver','guard_source','builder','fit_source'):verified(c[key])
    assert json.loads(verified(c['baseline_native8']).read_text())['admitted']
    roles=json.loads(verified(c['roles']).read_text());assert set(c['fit_ranks']+c['check_ranks'])<=set(roles['train'])
    assert not set(c['fit_ranks']+c['check_ranks'])&set(roles['valid']) and not set(c['fit_ranks'])&set(c['check_ranks'])
    base={};baseline_rows={};truth={}
    for b in c['controls']:
        m=json.loads(verified(b['manifest']).read_text());rows=[json.loads(s) for s in verified(b['rows']).read_text().splitlines()]
        g=json.loads(verified(b['guard']).read_text());assert m['sha256']==c['baseline']['sha256'] and m['candidate']==c['baseline']['path']
        assert m['ranks']==b['ranks']==[r['rank'] for r in rows]
        command=[c['runtime'],'-B',c['driver']['path'],c['baseline']['path'],','.join(map(str,b['ranks'])),Path(b['manifest']['path']).parent.name]
        checked_guard(g,command,150 if len(b['ranks'])==2 else 300,7500000000)
        for r in rows:
            if r['rank'] in c['fit_ranks']:baseline_rows[r['rank']]=r
        for r,p in b['predictions'].items():base[int(r)]=prediction(p)
    for item in c['data']:
        r=item['rank'];assert r in c['fit_ranks']
        for key in ('row_manifest','weights','truth'):verified(item[key])
        with np.load(item['truth']['path'],allow_pickle=False) as z:truth[r]=np.asarray(z['mc_post'][15],dtype=np.float64)
        assert truth[r].shape==(1024,) and np.isfinite(truth[r]).all()
        row=baseline_rows[r];assert row['util']==row['flops']/2**41
        assert math.isclose(float(np.mean((base[r][15]-truth[r])**2)),row['mse'],rel_tol=1e-12)
    arms={};arm_metrics=[]
    for a in c['arms']:
        verified(a['source']);folder=Path(a['output']);m=read_saved(folder/'manifest.json');g=read_saved(a['guard'])
        checked_guard(g,a['command'],a['limits']['seconds'],a['limits']['rss_bytes'])
        assert m['candidate']==a['source']['path'] and m['sha256']==a['source']['sha256'] and m['ranks']==c['fit_ranks']
        assert m['runtime']==c['runtime'] and m['numpy']=='2.4.6' and m['thread_env']==dict.fromkeys(THREADS,'1')
        assert 0<=m['setup']['setup_s']<=c['setup']['seconds_max']
        for k,v in c['setup']['counts'].items():assert [z[1] for z in m['setup']['pb_log'] if z[0]==k]==[v]
        rows=[json.loads(s) for s in verified(pin(folder/'rows.jsonl')).read_text().splitlines()]
        assert [r['rank'] for r in rows]==c['fit_ranks'];preds={}
        for row in rows:
            r=row['rank'];y=prediction(pin(folder/f'pred_{r}.npy'));assert y[0].tobytes()==base[r][0].tobytes()
            assert all(math.isfinite(row[k]) and row[k]>0 for k in ('mse','util','adjusted','cpu_s','wall_s'))
            assert 0<row['flops']<=2**41 and row['util']==row['flops']/2**41
            assert math.isclose(row['mse'],float(np.mean((y[15]-truth[r])**2)),rel_tol=1e-12)
            assert math.isclose(row['adjusted'],row['mse']*max(.1,row['util']),rel_tol=1e-14)
            u=row['ubc'];assert u['retired'] and u['retained_cache_bytes']==u['live_cache_bytes']==0 and u['max_resident_bytes']<=512000000
            preds[r]=y[15]
        arms[a['name']]=preds;arm_metrics.append(dict(name=a['name'],rows=rows))
    assert len(arms)==6
    AA=[];rr=[];diagnostics=[]
    for r in c['fit_ranks']:
        weight=max(.1,baseline_rows[r]['util']);scale=math.sqrt(weight/1024)
        odd=np.column_stack([(arms[k+'_plus'][r]-arms[k+'_minus'][r])/2 for k in c['dial_order']])
        even=np.column_stack([(arms[k+'_plus'][r]+arms[k+'_minus'][r])/2-base[r][15] for k in c['dial_order']])
        AA.append(odd*scale);rr.append((base[r][15]-truth[r])*scale)
        diagnostics.append(dict(rank=r,odd_weighted_squared=list(np.sum((odd*scale)**2,axis=0)),even_weighted_squared=list(np.sum((even*scale)**2,axis=0))))
    A=np.concatenate(AA);residual=np.concatenate(rr);G=A.T@A;g=A.T@residual
    assert np.isfinite(G).all() and np.isfinite(g).all()
    x,Q,feasible=box_solve(G,g);den=float(residual@residual);predicted=float((residual+A@x)@(residual+A@x))
    gain=100*(1-predicted/den);admitted=gain>=c['gates']['predicted_fit_gain_pct_min']
    values={k:c['baseline_values'][k]*(1+.02*float(x[i])) for i,k in enumerate(c['dial_order'])}
    report=dict(passed_integrity=True,endpoint_admitted=admitted,predicted_fit_adjusted_gain_pct=gain,x=x.tolist(),values=values,
        G=G.tolist(),g=g.tolist(),Q=Q.tolist(),feasible_active_sets=feasible,even_diagnostics=diagnostics,arm_metrics=arm_metrics,
        protocol=pin(D/'protocol.json'),evidence=list(EVIDENCE.values()),fit_ranks=c['fit_ranks'],check_targets_read=False,
        exact_gradient_claim=False,model_executed=False,endpoint=None)
    if admitted:
        parent=verified(c['baseline']).read_bytes()
        tail=('\n# One fixed native joint3 proposal; no endpoint or ridge search.\n'
            '_S19_BSCALE = '+repr(values['bscale'])+'\n'
            '_S19_LAM = (_S19_LAM[0], '+repr(values['newborn'])+')\n'
            '_K31R_TABLE = {l: (c[0], '+repr(values['c1'])+') for l, c in _K31R_TABLE.items()}\n').encode()
        with (D/'endpoint.py').open('xb') as f:f.write(parent+tail)
        assert (D/'endpoint.py').read_bytes()[:-len(tail)]==parent
        report['endpoint']=pin(D/'endpoint.py')
    once(D/'fit_result.json',report)
    print(json.dumps({k:report[k] for k in ('passed_integrity','endpoint_admitted','predicted_fit_adjusted_gain_pct','x','values','endpoint')}))

if __name__=='__main__':main()
