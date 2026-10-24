"""Fixed historical Thin-host mixed descriptor screen; no estimator or export."""
from pathlib import Path
import hashlib, json, math, os, sys

D = Path(__file__).resolve().parent
MINIS = [57, 68, 45, 59, 3, 12]
LAYERS = [10, 11, 12]
FIELDS = ['local21', 'local31', 'mixed21', 'mixed31']
THREADS = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
GATES = dict(covariance_ratio_at_most=.8, response_ratio_at_most=.8,
             covariance_network_wins_min=5, response_network_wins_min=5,
             covariance_extra_reduction_min=.02, response_extra_reduction_min=.02)


def pin(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1 << 20): h.update(b)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def once(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def main(protocol_path, digest):
    pp = pin(protocol_path); assert pp['sha256'] == digest
    p = json.loads(Path(protocol_path).read_text())
    assert p['screen'] == pin(__file__) and p['minis'] == MINIS and p['layers'] == LAYERS
    assert p['fields'] == FIELDS and p['gates'] == GATES and p['ridge_diagonal_fraction'] == .001
    assert p['host']['sha256'] == '25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee'
    assert p['stage'] == 0 and p['old_host_proxy_only'] and not p['native_candidate_authorized']
    assert str(Path(sys.executable).absolute()) == p['runtime']
    assert all(os.environ.get(k) == '1' for k in THREADS)
    assert not (D/'result.json').exists() and not (D/'sufficient_statistics.json').exists()
    # Verify each consumed file exactly once; packets are subsequently streamed.
    signatures = {}
    for record in [p['host'], *p['sources'], p['cell_manifest'], p['capture_manifest'], p['normalizers'],
                   *[r['packet'] for r in p['rows']], *p['raw_captures']]:
        key = record['path']
        if key in signatures: continue
        assert pin(key) == record, key
        st = Path(key).stat(); signatures[key] = (st.st_size, st.st_mtime_ns)
    manifest = json.loads(Path(p['cell_manifest']['path']).read_text())
    captures = json.loads(Path(p['capture_manifest']['path']).read_text())
    assert manifest['passed'] and manifest['stage'] == 0 and manifest['total_residual']
    assert not manifest['final_mean_targets_used'] and not manifest['privileged_innovation_labels_used']
    assert manifest['normalizers'] == p['normalizers'] and manifest['capture_manifest'] == p['capture_manifest']
    assert captures['passed'] and captures['stage'] == 0 and captures['policy'] == p['host']
    assert captures['completed'] and captures['all_processes_retired'] and not captures['targets_accessed']
    assert captures['all16_historical_thin_parity'] and captures['first_mini_fresh_plain_all16_and_bill_exact']
    actual = {(r['mini'], r['layer']): r for r in manifest['rows']}
    assert p['rows'] == [actual[m, l] for m in MINIS for l in LAYERS]
    assert p['raw_captures'] == [next(r['packet'] for r in captures['rows'] if r['mini'] == m) for m in MINIS]
    import numpy as np
    assert np.__version__ == '2.4.6'
    n = 1024; upper = np.triu_indices(n, 1)

    def read_arrays(record, keys):
        path = record['path']; st = Path(path).stat()
        assert (st.st_size, st.st_mtime_ns) == signatures[path]
        with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in keys}

    scales = read_arrays(p['normalizers'], ['rho_scale', 'third_scale'])
    assert all(np.isfinite(v).all() and np.all(v > 0) for v in scales.values())

    def project(A, m):
        h = m / np.sqrt(np.sum(m*m)); d = 1 - 2*h*h
        assert np.isfinite(h).all() and np.min(d) > 0
        A = (A+A.T)*.5; np.fill_diagonal(A, 0.)
        b = A@h; v = h/d; z = b/d
        u = 2*(z-v*np.dot(h,z)/(1+np.dot(h,v)))
        Q = A-.5*(np.outer(u,h)+np.outer(h,u)); np.fill_diagonal(Q, 0.)
        assert np.isfinite(Q).all()
        assert np.max(np.abs(Q@h)) <= 1e-10 * max(1., float(np.max(np.abs(Q))))
        return Q

    stats = []
    keys = ['R','S','sigma','m','W','T_upper','factor','response','energy','response_energy']
    for row in p['rows']:
        a = read_arrays(row['packet'], keys)
        F = read_arrays(row['capture'], [f"F_{row['layer']}"])[f"F_{row['layer']}"]
        assert F.shape == (n,94) and F.dtype == np.float32 and np.isfinite(F).all()
        assert all(a[k].shape == (n,n) and a[k].dtype == np.float32 for k in ('R','S','W'))
        assert all(a[k].shape == (n,) and a[k].dtype == np.float64 for k in ('sigma','m','factor','response'))
        assert a['T_upper'].shape == (n*(n-1)//2,) and a['T_upper'].dtype == np.float64
        assert all(np.isfinite(x).all() for x in a.values()) and np.min(a['sigma']) > 0
        # These are post-C/pre-sigma ratios and pre-K21 ratios, not a preactivation rho.
        R = a['R'] * scales['rho_scale']; S = a['S'] * scales['third_scale']
        np.fill_diagonal(R, 0.); np.fill_diagonal(S, 0.)
        V = S @ R  # One ordinary FP32 GEMM; an observable, not an exact cumulant identity.
        alpha, gate, pdf = (F[:,k].astype(np.float64) for k in (3,5,6))
        ss = np.outer(a['sigma'], a['sigma']); W = a['W'].astype(np.float64)
        triangle = []; responses = []
        for plane in (S, V):
            for weight, divisor in ((pdf,2.),(-alpha*pdf,6.)):
                A = plane.astype(np.float64) * np.outer(weight,gate)
                B = project(ss*(A+A.T)/divisor, a['m'])
                triangle.append(B[upper].copy())
                responses.append(a['factor'] * np.sum(W*(B@W), axis=0))
                del A, B
        X = np.stack(triangle); Y = np.stack(responses)
        tc = a['T_upper']; tr = a['response']
        cg = (2./(n*n))*(X@X.T); ch = (2./(n*n))*(X@tc); ce = (2./(n*n))*np.dot(tc,tc)
        rg = (Y@Y.T)/n; rh = (Y@tr)/n; re = np.dot(tr,tr)/n
        assert ce > 0 and re > 0 and np.isclose(ce,float(a['energy']),rtol=1e-10,atol=0.)
        assert np.isclose(re,float(a['response_energy']),rtol=1e-10,atol=0.)
        assert all(np.isfinite(x).all() for x in (cg,ch,ce,rg,rh,re))
        stats.append(dict(mini=row['mini'],layer=row['layer'],covariance_gram=cg.tolist(),
             covariance_dot=ch.tolist(),covariance_energy=float(ce),response_gram=rg.tolist(),
             response_dot=rh.tolist(),response_energy=float(re)))
        print(json.dumps(dict(cell_complete=[row['mini'],row['layer']]),allow_nan=False),flush=True)
        del a,F,R,S,V,ss,W,triangle,responses,X,Y,tc,tr,cg,ch,rg,rh
    sp = D/'sufficient_statistics.json'
    once(sp,dict(protocol=pp,screen=p['screen'],fields=FIELDS,rows=stats,
                 covariance_uses_strict_upper_times_two=True,old_host_proxy_only=True))

    def matrices(row, count):
        C=np.asarray(row['covariance_gram'])[:count,:count]; c=np.asarray(row['covariance_dot'])[:count]
        R=np.asarray(row['response_gram'])[:count,:count]; r=np.asarray(row['response_dot'])[:count]
        return C,c,R,r,row['covariance_energy'],row['response_energy']

    output = {}
    for name,count in [('local',2),('augmented',4)]:
        folds = []; cov_all = []; response_all = []
        for held in MINIS:
            train = [r for r in stats if r['mini'] != held]
            G=np.zeros((count,count)); h=np.zeros(count)
            for row in train:
                C,c,R,r,ce,re=matrices(row,count)
                G += .5*(C/ce+R/re); h += .5*(c/ce+r/re)
            G/=len(train); h/=len(train)
            scale=np.sqrt(np.diag(G)); assert np.isfinite(scale).all() and np.all(scale>0)
            beta=np.linalg.solve(G/np.outer(scale,scale)+.001*np.eye(count),h/scale)/scale
            assert np.isfinite(beta).all()
            cells=[]
            for row in stats:
                if row['mini'] != held: continue
                C,c,R,r,ce,re=matrices(row,count)
                cr=float((ce-2*np.dot(c,beta)+beta@C@beta)/ce)
                rr=float((re-2*np.dot(r,beta)+beta@R@beta)/re)
                assert math.isfinite(cr) and math.isfinite(rr) and cr>=0 and rr>=0
                cells.append(dict(layer=row['layer'],covariance_ratio=cr,response_ratio=rr))
                cov_all.append(cr); response_all.append(rr)
            folds.append(dict(held_mini=held,training_minis=[m for m in MINIS if m!=held],
                 coefficients=beta.tolist(),covariance_ratio=sum(x['covariance_ratio'] for x in cells)/3,
                 response_ratio=sum(x['response_ratio'] for x in cells)/3,cells=cells))
        output[name]=dict(folds=folds,covariance_ratio=sum(cov_all)/18,response_ratio=sum(response_all)/18,
             covariance_network_wins=sum(f['covariance_ratio']<1 for f in folds),
             response_network_wins=sum(f['response_ratio']<1 for f in folds))
    a=output['augmented']; b=output['local']
    gates=dict(covariance_ratio=a['covariance_ratio']<=.8,response_ratio=a['response_ratio']<=.8,
         covariance_network_wins=a['covariance_network_wins']>=5,response_network_wins=a['response_network_wins']>=5,
         covariance_extra_reduction=b['covariance_ratio']-a['covariance_ratio']>=.02,
         response_extra_reduction=b['response_ratio']-a['response_ratio']>=.02)
    result=dict(passed=all(gates.values()),passed_integrity=True,protocol=pp,screen=p['screen'],
         sufficient_statistics=pin(sp),gates=gates,metrics=output,all_finite=True,old_host_proxy_only=True,
         estimator_imported=False,native_candidate_exported=False,full_data_fit_performed=False,
         final_mean_targets_used=False,hyperparameter_or_layer_selection=False)
    once(D/'result.json',result)
    print(json.dumps(dict(passed=result['passed'],gates=gates,
                         local={k:v for k,v in b.items() if k!='folds'},
                         augmented={k:v for k,v in a.items() if k!='folds'}),allow_nan=False),flush=True)


if __name__ == '__main__':
    assert len(sys.argv)==3
    main(sys.argv[1],sys.argv[2])
