"""One frozen first128 on-policy second step; conditional five-fold TRAIN72 CV.

All four existing head slots receive additions rounded to float32. All five
coefficient seals precede validation reporting. Full90 fits only after gates.
No export, native execution, model selection, grid, or automatic retry.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, gzip, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
D = HERE.parent/'truth_response95_v1'
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
OUT = RUNS/'onpolicy_first128_second_fit_v1'
PREREG = '42c4f80e65348f149e4b64d8e91da29f0fb1995abb5a78f8004fe0623a4a5bd9'
GUARD = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
BUILDER = '3ca07f38b1799ec03bf45705aec3344dbf108ac51a502ab917839752d2d3db4e'
N, WIDTH = 1024, 321
CHANGED = ('head', 'head_b', 'hidden_head', 'first_hidden_head')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
assert sha(HERE/'build_parent_sources_v1.py') == BUILDER
spec = importlib.util.spec_from_file_location('op1_frozen_parent_builder', HERE/'build_parent_sources_v1.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
cf, f = builder.cf, builder.f
assert (cf.RIDGE, cf.DAMPING, cf.PROJECTION_RIDGE) == (.1, .5, 1e-10)

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def guardcheck(path, threads):
    g = read(path)
    assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0 and not g['cleanup_errors']
    assert g['guard_sha256'] == GUARD and g['thread_provenance']['blas_threads'] == threads
    assert g['cap_bytes'] == 6500000000 and g['reserve_bytes'] == 19000000000 and g['max_active'] == 36
    assert g['max_owned_swap_bytes'] == 0 and all(v == 0 for v in g['cgroup_events_final'].values())
    return sha(path)

def frozen(c, head, bias, hidden, first):
    q = c.copy()
    q[:64] = (head.astype('f8')+q[:64]).astype('f4').astype('f8')-head.astype('f8')
    q[64] = float(np.float32(float(bias)+q[64]))-float(bias)
    q[65:193] = (hidden.astype('f8')+q[65:193]).astype('f4').astype('f8')-hidden.astype('f8')
    q[193:] = (first.astype('f8')+q[193:]).astype('f4').astype('f8')-first.astype('f8')
    assert np.isfinite(q).all()
    return q

def one_layer(G, b, n, head, bias, hidden, first, U):
    # Same original state64 and hidden-null64 sequence, with existing-head addition.
    baseline = np.zeros(WIDTH)
    baseline[:64] = cf.solve(G[:64,:64], b[:64], n)
    baseline = frozen(baseline, head, bias, hidden, first)
    T0 = np.zeros((193,64)); T0[65:] = U
    gg, cross = G[:65,:65], G[:65,:193]@T0
    scale = np.maximum(np.sqrt(np.maximum(np.diag(gg),0)/n),1e-8)
    A = np.linalg.solve(gg/(scale[:,None]*scale[None,:])+1e-10*n*np.eye(65), cross/scale[:,None])/scale[:,None]
    R0 = T0.copy(); R0[:65] -= A
    gres = R0.T@G[:193,:193]@R0; gres = .5*(gres+gres.T)
    k = cf.solve(gres, R0.T@(b[:193]-G[:193,:193]@baseline[:193]), n)
    baseline[:193] += R0@k
    baseline = frozen(baseline, head, bias, hidden, first)
    # First128 residualized against the same state65+null64 independent span.
    Q = np.zeros((WIDTH,129)); Q[:65,:65] = np.eye(65); Q[65:193,65:] = U
    T = np.zeros((WIDTH,128)); T[193:] = np.eye(128)
    gg, cross = Q.T@G@Q, Q.T@G@T
    scale = np.maximum(np.sqrt(np.maximum(np.diag(gg),0)/n),1e-8)
    A = np.linalg.solve(gg/(scale[:,None]*scale[None,:])+1e-10*n*np.eye(129), cross/scale[:,None])/scale[:,None]
    R = T-Q@A
    gres = R.T@G@R; gres = .5*(gres+gres.T)
    k = cf.solve(gres, R.T@(b-G@baseline), n)
    return frozen(baseline+R@k, head, bias, hidden, first)

def coefficients(G, b, nnet, P):
    U = cf.nullspace(P['enc4_w'])
    return np.stack([one_layer(G[l],b[l],nnet*N,P['head'][l+1],P['head_b'][l+1],
                              P['hidden_head'][l+1],P['first_hidden_head'][l+1],U) for l in range(15)])

def deploy(P, coef):
    assert coef.shape == (15,321)
    out = {k:v.copy() for k,v in P.items()}
    for key, columns in (('head',slice(0,64)),('head_b',64),('hidden_head',slice(65,193)),('first_hidden_head',slice(193,321))):
        out[key][1:] = (P[key][1:].astype('f8')+coef[:,columns]).astype('f4')
    assert set(out) == set(P)
    for key in P:
        assert P[key].shape == out[key].shape and P[key].dtype == out[key].dtype
        if key not in CHANGED: np.testing.assert_array_equal(out[key],P[key])
        else: np.testing.assert_array_equal(out[key][0],P[key][0])
    return out

def tensor_digest(P):
    h = hashlib.sha256()
    for key in sorted(P):
        h.update(key.encode()+str(P[key].shape).encode()+P[key].dtype.str.encode()+P[key].tobytes())
    return h.hexdigest()

def delta32(P, layer, state, hidden, first, sigma):
    # Exact49bd expression order. Only coefficient bytes differ in deployed source.
    ss, hh, ff, sig = [np.asarray(v,dtype='f4') for v in (state,hidden,first,sigma)]
    inner = ((ss@P['head'][layer]+hh@P['hidden_head'][layer])+ff@P['first_hidden_head'][layer])+P['head_b'][layer]
    gain = np.float32(1)+np.float32(.25)*np.tanh(P['gain'][layer:layer+1])
    result = ((sig*P['rms'][layer])*inner)*gain
    assert result.dtype == np.dtype('f4') and np.isfinite(result).all()
    return result

def custody(recipe):
    assert sha(HERE/'PREREG_v1.json') == PREREG
    manifest_sha = recipe['source_manifest_sha256']
    assert sha(HERE/'PARENT_SOURCES_v1.json') == manifest_sha
    pre, manifest = read(HERE/'PREREG_v1.json'), read(HERE/'PARENT_SOURCES_v1.json')
    assert manifest['successful'] and manifest['full90_exact_49bd_tensor_parity'] and manifest['all_new_adaptation_heads_fold_fit']
    ready_path, final_path = HERE/'capture_plans_v1/CAPTURE_READY_v1.json', HERE/'CAPTURE_FINAL_v1.json'
    assert sha(ready_path) == recipe['capture_ready_sha256']
    ready, final = read(ready_path), read(final_path)
    assert final['successful'] and final['captured_predictions'] == 540 and final['unhooked_controls'] == 18
    assert final['ready_sha256'] == sha(ready_path) and final['prereg_sha256'] == PREREG
    assert final['source_manifest_sha256'] == manifest_sha and final['no_truth_arrays_read']
    assert final['all_six_prediction_and_charged_invoice_anchors_exact']
    assert sha(D/'DESIGN_LOCAL_v1.json') == pre['design_sha256']
    design = read(D/'DESIGN_LOCAL_v1.json')
    assert design['train'] == pre['train'] and design['weight_hash_disjointness']
    assert not set(pre['train'])&set(design['protected_seed_ids'])
    records, proofs = {}, []
    assert len(manifest['families']) == len(ready['outputs']) == 6
    for index,(family,capture) in enumerate(zip(manifest['families'],ready['outputs'])):
        key = family['key']; held = pre['train'][index::5] if index<5 else []
        assert key == ('fold%d'%index if index<5 else 'full90') and capture['key'] == key
        assert family['validation'] == held and family['train'] == [s for s in pre['train'] if s not in held]
        assert family['parent_fit_target_seed_ids'] == family['train'] and family['parent_fit_excludes_validation']
        assert sha(ROOT/family['coef_path']) == family['coef_sha256']
        assert sha(ROOT/family['build_path']) == family['build_sha256']
        assert capture['teacher'] == family['science']
        source = ROOT/family['science']['path']; assert sha(source) == family['science']['sha256']
        if index == 5: assert sha(source) == pre['parent_science_sha256']
        files, predictions, stages = {}, {}, []
        for stage in ('anchor','remaining'):
            item = capture['outputs'][stage]; folder = Path(item['out']); audit = folder/'AUDIT_v1.json'
            proof = final['families'][key][stage]
            assert sha(audit) == proof['audit_sha256']
            assert guardcheck(Path(item['audit_guard_path']),1) == proof['audit_guard_sha256']
            ar = read(audit)
            assert ar['successful'] and ar['family'] == key and ar['stage'] == stage
            assert ar['source_sha256'] == family['science']['sha256'] and ar['ready_sha256'] == sha(ready_path)
            assert ar['source_manifest_sha256'] == manifest_sha and ar['prereg_sha256'] == PREREG
            assert ar['auditor_sha256'] == recipe['capture_auditor_sha256'] and ar['no_truth_arrays_read']
            assert ar['all_schema_hash_finite_checks_pass'] and ar['current_hidden_and_first_heads_in_replay']
            if stage == 'anchor':
                assert ar['all3_anchor_predictions_byte_identical'] and ar['all3_anchor_charged_invoices_exact']
                assert len(ar['prediction_pairs']) == 3
            else: assert ar['remaining_uses_same_source_anchor_proof']
            for name in ('plan','recipe'):
                assert sha(ROOT/item[name]['path']) == item[name]['sha256'] == ar[name+'_sha256']
            for row in ar['guards']: assert guardcheck(Path(row['path']),4) == row['sha256']
            for row in ar['files']:
                p = Path(row['path']); assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
                ident = row['seed'],row['layer']; assert ident not in files; files[ident] = row
            for row in ar['predictions']:
                p = Path(row['path']); assert p.stat().st_size == 65536 and sha(p) == row['sha256']
                assert row['seed'] not in predictions; predictions[row['seed']] = row
            stages.append(dict(stage=stage,audit_sha256=sha(audit),guard_sha256=proof['audit_guard_sha256']))
        assert set(predictions) == set(pre['train']) and set(files) == {(s,l) for s in pre['train'] for l in range(1,16)}
        lit,_ = f.literals_and_functions(source.read_text()); P = f.decode(lit['_GNN_LITERAL'])
        assert P['first_hidden_head'].shape == P['hidden_head'].shape == (16,128)
        records[key] = dict(family=family,files=files,predictions=predictions,P=P)
        proofs.append(dict(family=key,source_sha256=sha(source),stages=stages,all90_files_verified=True))
    return pre,design,records,dict(capture_final_sha256=sha(final_path),families=proofs,all540_before_truth_array_open=True)

def data_rows(seeds, info, design, plan, role, seal=None):
    family = info['family']; P = info['P']
    assert role in ('TRAIN','HELD')
    assert list(seeds) == family['train' if role=='TRAIN' else 'validation']
    assert not set(seeds)&set(design['protected_seed_ids'])
    if role == 'HELD':
        assert seal is not None and read(seal)['family'] == family['key']
        assert read(seal)['excluded_validation'] == list(seeds)
    for seed in seeds:
        row, wr = design['rows'][str(seed)], plan['rows'][str(seed)]['weights']
        wp, tp = ROOT/wr['path'], ROOT/row['truth']['path']
        assert wp.stat().st_size == wr['bytes'] and sha(wp) == wr['sha256']
        assert row['raw_float32_sha256'] == wr['raw_float32_sha256'] and wr['raw_float32_sha256'] not in design['protected_raw_weight_hashes']
        assert tp.stat().st_size == row['truth']['bytes'] and sha(tp) == row['truth']['sha256']
        W = np.load(wp,mmap_mode='r',allow_pickle=False); assert W.shape == (16,N,N)
        pred = np.fromfile(info['predictions'][seed]['path'],dtype='<f4').reshape(16,N).astype('f8')
        with np.load(tp,allow_pickle=False) as truth:
            assert int(truth['seed']) == seed and str(truth['weights_sha256']) == wr['raw_float32_sha256']
            A, B = truth['mc_post_A'].astype('f8')-pred, truth['mc_post_B'].astype('f8')-pred
        assert A.shape == B.shape == (16,N) and np.isfinite(A).all() and np.isfinite(B).all()
        for layer in range(1,16):
            path = Path(info['files'][seed,layer]['path']); assert sha(path) == info['files'][seed,layer]['sha256']
            with gzip.open(path,'rt') as stream: cap = json.load(stream)
            assert (cap['rank'],cap['phase'],cap['layer']) == (seed,'scored',layer)
            state,hidden,first,sigma,ng,saved = [cf.arr(cap,k) for k in ('state','hidden2','first128','sigma','gate_feature','delta')]
            assert state.shape == (N,64) and hidden.shape == first.shape == (N,128)
            assert sigma.shape == ng.shape == saved.shape == (N,) and np.max(np.abs(ng)) < 7.999
            gate = ng*float(P['scales'][layer,5])+float(P['centers'][layer,5])
            assert gate.min()>-2e-6 and gate.max()<1+2e-6; gate=np.clip(gate,0,1)
            physical = sigma*float(P['rms'][layer])*(1+.25*np.tanh(float(P['gain'][layer])))
            assert physical.min()>1e-12
            old = delta32(P,layer,state,hidden,first,sigma)
            assert np.max(np.abs(old.astype('f8')-saved)) < 1e-8+2e-5*np.max(np.abs(saved))
            assert np.max(np.abs(state-(hidden@P['enc4_w'].astype('f8').T+P['enc4_b'].astype('f8')))) < 3e-5
            mask = (pred[layer]>1e-12).astype('f8') if layer<15 else np.ones(N)
            X = np.column_stack([state,np.ones(N),hidden,first])*mask[:,None]
            inherited = np.asarray(W[layer],dtype='f8').T@np.stack([A[layer-1],B[layer-1]],axis=1)
            yA,yB = (A[layer]-gate*inherited[:,0])/physical,(B[layer]-gate*inherited[:,1])/physical
            assert np.isfinite(yA).all() and np.isfinite(yB).all()
            for value in (state,hidden,first,sigma): value.setflags(write=False)
            yield dict(seed=seed,layer=layer,state=state,hidden=hidden,first=first,sigma=sigma,physical=physical,
                       saved=saved,old=old,mask=mask,X=X,yA=yA,yB=yB)
        del W
        print(json.dumps(dict(utc=utc(),family=family['key'],role=role,loaded_seed=seed)),flush=True)

def moments(rows, seeds):
    G,b = np.zeros((15,321,321)),np.zeros((15,321))
    loaded=[]
    for r in rows:
        l = r['layer']-1; G[l] += r['X'].T@r['X']; b[l] += r['X'].T@(.5*(r['yA']+r['yB']))
        if r['layer']==15: loaded.append(r['seed'])
    assert loaded == list(seeds)
    return G,b

def evaluate(rows, P, Q, seeds):
    result=[]; numerator=np.zeros(15); denominator=np.zeros(15); previous=None; original=loss=0.0
    for r in rows:
        seed,layer=r['seed'],r['layer']; l=layer-1
        if seed!=previous:
            assert layer==1
            previous,original,loss=seed,0.0,0.0
        before = [hashlib.sha256(r[k].tobytes()).hexdigest() for k in ('state','hidden','first','sigma')]
        new = delta32(Q,layer,r['state'],r['hidden'],r['first'],r['sigma'])
        change = (new.astype('f8')-r['old'].astype('f8'))*r['mask']
        correction=change/r['physical']
        original += float(r['yA']@r['yB'])
        loss += float((r['yA']-correction)@(r['yB']-correction))
        numerator[l] += float(change@change)
        masked=r['saved']*r['mask']; denominator[l] += float(masked@masked)
        assert before == [hashlib.sha256(r[k].tobytes()).hexdigest() for k in ('state','hidden','first','sigma')]
        if layer==15:
            assert original>0 and loss>0
            result.append(dict(seed=seed,original=original/(15*N),baseline=original/(15*N),candidate=loss/(15*N),
                               incremental_pct=100*(loss/original-1),total_pct=100*(loss/original-1),baseline_pct=0.0,
                               direct_float32_parent_and_candidate_delta_replay=True))
    assert [r['seed'] for r in result] == list(seeds) and np.all(denominator>0)
    ratios=np.sqrt(numerator/denominator);overall=float(np.sqrt(numerator.sum()/denominator.sum()))
    trust=dict(correction_to_parent_head_rms=overall,per_layer_ratios=ratios.tolist(),max_layer_ratio=float(ratios.max()),
               passed=bool(overall<=.25 and ratios.max()<=1),physical_actual_float32_delta_difference=True)
    return result,trust

def selftest():
    original=cf.selftest();rng=np.random.default_rng(202610211001)
    E=(rng.normal(size=(64,128))*.1).astype('f4'); U=cf.nullspace(E)
    hidden=rng.normal(size=(703,128));state=hidden@E.T;first=rng.normal(size=(703,128));X=np.column_stack([state,np.ones(703),hidden,first])
    y=rng.normal(size=703);G,b=X.T@X,X.T@y;head=(rng.normal(size=64)*.01).astype('f4');bias=np.float32(.01);zero=np.zeros(128,'f4')
    expected=cf.one_layer(G,b,len(X),head,bias,U)[1]
    np.testing.assert_array_equal(one_layer(G,b,len(X),head,bias,zero,zero,U),expected)
    hh=(rng.normal(size=128)*.01).astype('f4');ff=(rng.normal(size=128)*.01).astype('f4')
    c=one_layer(G,b,len(X),head,bias,hh,ff,U)
    np.testing.assert_array_equal(c,frozen(c,head,bias,hh,ff))
    selected=np.arange(0,len(X),5);excluded=np.setdiff1d(np.arange(len(X)),selected)
    c0=one_layer(X[selected].T@X[selected],X[selected].T@y[selected],len(selected),head,bias,hh,ff,U)
    X[excluded]+=1e6;y[excluded]+=1e9
    c1=one_layer(X[selected].T@X[selected],X[selected].T@y[selected],len(selected),head,bias,hh,ff,U)
    np.testing.assert_array_equal(c0,c1)
    P=dict(head=np.tile(head,(16,1)),head_b=np.full(16,bias,'f4'),hidden_head=np.tile(hh,(16,1)),first_hidden_head=np.tile(ff,(16,1)),
           enc4_w=E,enc4_b=np.zeros(64,'f4'),rms=np.full(16,.0001,'f4'),gain=np.zeros(16,'f4'))
    Q=deploy(P,np.tile(c,(15,1)));Z=deploy(P,np.zeros((15,321)))
    assert tensor_digest(P)==tensor_digest(Z)
    ss,hs,fs,sig=[np.asarray(v,'f4') for v in (state,hidden,first,rng.uniform(.2,1.2,len(state)))]
    old=delta32(P,5,ss,hs,fs,sig);new=delta32(Q,5,ss,hs,fs,sig)
    np.testing.assert_array_equal(old,delta32(Z,5,ss,hs,fs,sig))
    exact=((sig*P['rms'][5])*(((ss@P['head'][5]+hs@P['hidden_head'][5])+fs@P['first_hidden_head'][5])+P['head_b'][5]))*(np.float32(1)+np.float32(.25)*np.tanh(P['gain'][5:6]))
    np.testing.assert_array_equal(old,exact)
    mask=(np.arange(len(old))%3!=0).astype('f8');change=(new.astype('f8')-old.astype('f8'))*mask
    assert np.all(change[mask==0]==0)
    a,bv=rng.normal(size=len(old)),rng.normal(size=len(old));corr=change/.0001
    np.testing.assert_allclose((a-corr)@(bv-corr),a@bv-corr@(a+bv)+corr@corr,atol=1e-10,rtol=1e-10)
    return dict(original_first128=original,zero_prior_hidden_first_matches_original_exact=True,all_four_existing_heads_additive_float32=True,
                excluded_features_labels_invariant=True,zero_correction_all_tensor_bytes_exact=True,original_delta_float32_order=True,
                only_existing_head_shapes_change=True,clip_mask_preserved=True,direct_cross_half_metric_identity=True,
                frozen_ridge_damp_projection=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    checks=selftest()
    if args.self_test: print(json.dumps(dict(successful=True,proofs=checks)),flush=True);return
    assert sys.platform=='linux'
    recipe=read(HERE/'SECOND_FIT_RECIPE_v1.json');review=read(HERE/'ROOT_SECOND_FIT_REVIEW_v1.json')
    assert recipe['fitter_sha256']==sha(__file__) and review['successful'] and review['one_fixed_fit_authorized']
    assert review['recipe_sha256']==sha(HERE/'SECOND_FIT_RECIPE_v1.json')
    pre,design,all_info,proof=custody(recipe)
    pp=D/'PLAN_CAPTURE_v1.json';assert sha(pp)=='a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375';plan=read(pp)
    OUT.mkdir(exist_ok=False);reports=[];models={};seals={};validation_rows=[]
    for key in ('fold0','fold1','fold2','fold3','fold4'):
        info=all_info[key];selected=info['family']['train'];P=info['P']
        G,b=moments(data_rows(selected,info,design,plan,'TRAIN'),selected)
        coef=coefficients(G,b,len(selected),P);Q=deploy(P,coef);del G,b
        rows,trust=evaluate(data_rows(selected,info,design,plan,'TRAIN'),P,Q,selected);trust['training_only']=True
        seal=OUT/(key+'_COEFFICIENTS_BEFORE_VALIDATION_v1.json')
        write(seal,dict(utc=utc(),family=key,train=selected,excluded_validation=info['family']['validation'],coef_by_layer=coef.tolist(),
                        physical=trust,parent_tensor_digest=tensor_digest(P),candidate_tensor_digest=tensor_digest(Q),
                        source_sha256=info['family']['science']['sha256'],prereg_sha256=PREREG,explicit_train_sum_only=True,
                        no_own_validation_truth_used_for_this_fit=True,shared_historical_prior_conditional=True))
        models[key],seals[key]=Q,seal
        reports.append(dict(family=key,training=cf.summary(rows),physical=trust,coefficient_seal_sha256=sha(seal)))
    seal_record=OUT/'ALL_FIVE_SEALED_BEFORE_VALIDATION_v1.json'
    write(seal_record,dict(utc=utc(),seals={k:sha(p) for k,p in seals.items()},all_five_sealed_before_any_validation_report=True))
    for report in reports:
        key=report['family'];info=all_info[key];held=info['family']['validation'];seal=seals[key]
        assert sha(seal)==report['coefficient_seal_sha256']
        rows,_=evaluate(data_rows(held,info,design,plan,'HELD',seal),info['P'],models[key],held)
        rows=[dict(r,fold=key,sealed_coef_sha256=sha(seal)) for r in rows]
        validation_rows.extend(rows);report['validation']=cf.summary(rows);report['validation_seed_ids']=held
    assert len(validation_rows)==90 and len({r['seed'] for r in validation_rows})==90 and {r['seed'] for r in validation_rows}==set(pre['train'])
    cv=cf.summary(validation_rows);negative=sum(r['validation']['mean_incremental_relative_pct']<0 for r in reports)
    cv_pass=bool(cv['mean_incremental_relative_pct']<=-.15 and cv['incrementally_lower']>=54 and negative>=4 and all(r['physical']['passed'] for r in reports))
    final_coef=None;full_report=None
    if cv_pass:
        info=all_info['full90'];selected=info['family']['train'];P=info['P']
        G,b=moments(data_rows(selected,info,design,plan,'TRAIN'),selected);coef=coefficients(G,b,len(selected),P);Q=deploy(P,coef);del G,b
        rows,trust=evaluate(data_rows(selected,info,design,plan,'TRAIN'),P,Q,selected);trust['training_only']=True
        seal=OUT/'full90_COEFFICIENTS_v1.json'
        write(seal,dict(utc=utc(),family='full90',train=selected,coef_by_layer=coef.tolist(),physical=trust,all_cv_gates_previously_passed=True,
                        parent_tensor_digest=tensor_digest(P),candidate_tensor_digest=tensor_digest(Q),prereg_sha256=PREREG))
        full_report=dict(training=cf.summary(rows),physical=trust,coefficient_seal_sha256=sha(seal));final_coef=coef.tolist()
    passed=bool(cv_pass and full_report['physical']['passed'])
    result=dict(utc=utc(),successful=True,passed=passed,status='FIVE_FOLD_FIRST128_ONPOLICY_PASS_NATIVE_REQUIRED' if passed else 'CLOSED_FIRST128_ONPOLICY_FROZEN_GATES_FAIL',
                script_sha256=sha(__file__),prereg_sha256=PREREG,source_manifest_sha256=recipe['source_manifest_sha256'],capture_custody=proof,
                selftests=checks,cross_validation=cv,negative_blocks=negative,cv_per_network=validation_rows,families=reports,
                all_five_seal_record_sha256=sha(seal_record),full90_fit_performed=cv_pass,full90=full_report,final_coef_by_layer=final_coef,
                shared_historical777a_prior_conditional_not_fully_nested=True,no_protected_truth_read=True,all_targets_authentic_TRAIN90_only=True,
                direct_f32_parent_candidate_head_replay=True,coefficients_sealed_before_validation=True,added_runtime_operations=0,
                no_adjusted_gain_claim=True,no_candidate_export=True,no_native_launch=True,no_grid_or_repeat=True,
                limits=['Local fixed-input proxy omits nonlinear trajectory feedback; own final recurrent validation/bills/runtime still required.',
                        'Frozen historical encoders/context/prior are shared; only all new adaptation heads have strict TRAIN72 exclusion.',
                        'Current paid49bd runtime debit persists; no source-capture timing is runtime evidence.'])
    write(OUT/'FITTED_v1.json',result)
    print(json.dumps({k:result[k] for k in ('utc','status','cross_validation','negative_blocks','full90_fit_performed')}),flush=True)

if __name__=='__main__': main()
