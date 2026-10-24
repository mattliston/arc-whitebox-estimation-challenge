import sys,types,importlib.util,time,resource,gc
from common import HERE,D,P2,A,TCI,PARENT,sha,pin,read,write,verify,basis,project

def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify();assert read(HERE/'unit.json')['passed']
    guard=read(HERE/'unit_watch.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
    oldguard=read(TCI/'capture_watch.json');assert oldguard['completed'] and oldguard['whole_group_retired'] and oldguard['max_swap_growth_bytes']==0
    cap=read(TCI/'capture/result.json');assert cap['completed'] and cap['repeat_all_captured_arrays_exact']
    started=time.monotonic();stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('tcs_parent',PARENT);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as setup:
        spec.loader.exec_module(mod)
        arena=fnp.empty((mod._PACKED_ELEMENTS,),dtype=fnp.float32)
        mod._SCRATCH=[arena,mod._cp3_bind_arena_views(arena,mod._PACKED_PUSH),mod._cp3_bind_arena_views(arena,mod._PACKED_READ),None,None,None,None]
        mod._bind_push6(arena)
    replay_rows=[];outputs={};records=[];truth_means={};zero_tolerance=True
    def native(a,C,label):
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            W=fnp.asarray(a['W']);CC=fnp.asarray(C);H=fnp.asarray(a['H14'].copy())
            P={k[2:]:fnp.asarray(v.copy()) for k,v in a.items() if k.startswith('P_')}
            v,av=mod._cov_compact_diagonal(W,CC,fnp.asarray(a['indices']),fnp.asarray(a['alive']))
            k3=fnp.asarray(a['k3base'])+3*av*fnp.asarray(a['bpre'])
            m,aux=mod.nonlinear_scalar(fnp.asarray(a['mu']),v,k3,fnp.asarray(a['radial'])*v*v)
            x=mod.final_scalar_features(15,m,aux,W)
            delta,_=mod.gnn_step(15,x,aux['sigma'],W,H,P)
            ans=fnp.asarray(fnp.maximum(m+fnp.asarray(delta,dtype=fnp.float64),0),dtype=fnp.float32)
        y=np.asarray(ans).copy();vv=np.asarray(v).copy();avv=np.asarray(av).copy()
        assert np.isfinite(y).all() and np.isfinite(vv).all() and np.all(vv>=1e-10),label
        row=dict(label=label,flops=ctx.summary_dict()['flops_used'],min_pre_variance=float(vv.min()),output_sha256=__import__('hashlib').sha256(y.tobytes()).hexdigest())
        replay_rows.append(row)
        return y,vv,avv,row
    def energy(xs):
        observed=float(np.mean([np.mean(x*x) for x in xs]));r=dict(observed_mse=observed)
        if len(xs)==2:
            cross=float(np.mean(xs[0]*xs[1]));noise=float(np.mean((xs[0]-xs[1])**2)/2)
            assert abs(observed-cross-noise)<=1e-12*max(observed,1e-30)
            r.update(cross_mse=cross,noise_mse=noise)
        return r
    for rank,call in [(35,0),(100,1)]:
        rec=cap['records'][call];assert sha(rec['capture'])==rec['capture_sha256']
        with np.load(rec['capture'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
        I=a['indices'];alive=a['alive'].astype(bool);cold=~alive;C=a['C'].astype(np.float64);W=a['W'].astype(np.float64)
        assert set(I.tolist())==set(np.flatnonzero(alive).tolist())
        Q,s,rms=basis(C,a['H14'],I);orth=float(np.linalg.norm(Q.T@Q-np.eye(Q.shape[1])));assert orth<=1e-10
        outputs[f'r{rank}_Q']=Q;outputs[f'r{rank}_singular_values']=s;outputs[f'r{rank}_column_rms']=rms
        outputs[f'r{rank}_parent']=a['prediction'][-1].copy()
        zero,_,_,zr=native(a,C,f'{rank}_zero');diff=zero.astype(np.float64)-a['prediction'][-1]
        zm=float(np.max(np.abs(diff)));zrm=float(np.sqrt(np.mean(diff*diff)))
        assert zm<=1e-6 and zrm<=1e-7,(rank,zm,zrm)
        paths=[P2/f'public_g19_attribution_v1/mps_cov2_rank{rank:04d}.npz']
        if rank==35:paths.append(A/'covariance_replica_truth_v1/rank_0035_8M_v1.npz')
        er=[];res=[];varerr=[];varres=[];stream_rows=[];truth_means[rank]=[]
        for stream,path in enumerate(paths):
            with np.load(path,allow_pickle=False) as z:
                assert int(z['P'])==8000000
                trueC=z['mc_post_cov'][14].astype(np.float64);truem=z['mc_post'][14].astype(np.float64)
                truefinal=z['mc_post'][15].astype(np.float64);preC=z['mc_pre_cov'][15].astype(np.float64);prem=z['mc_pre_m'][15].astype(np.float64)
            mr=float(np.linalg.norm(W.T@truem-prem)/np.linalg.norm(prem));cr=float(np.linalg.norm(W.T@trueC@W-preC)/np.linalg.norm(preC))
            assert max(mr,cr)<=1e-4,(mr,cr)
            E=trueC[np.ix_(I,I)]-C[np.ix_(I,I)];E=(E+E.T)*.5
            Ep,theta=project(E,Q);residual=E-Ep
            residual_orth=float(np.linalg.norm(Q.T@residual@Q)/max(np.linalg.norm(E),1e-30));assert residual_orth<=1e-10
            changed=C.copy();changed[np.ix_(I,I)]+=Ep
            full=np.zeros_like(C);full[np.ix_(I,I)]=trueC[np.ix_(I,I)];np.fill_diagonal(full,np.diag(trueC))
            yp,vp,ap,pr=native(a,changed,f'{rank}_{stream}_projected')
            yf,vf,af,fr=native(a,full,f'{rank}_{stream}_full')
            if rank==35 and stream==0:
                again,vr,ar,rr=native(a,changed,'35_0_projected_repeat')
                assert np.array_equal(yp,again) and np.array_equal(vp,vr) and np.array_equal(ap,ar) and pr['flops']==rr['flops']
            outputs[f'r{rank}_{stream}_projected']=yp;outputs[f'r{rank}_{stream}_full']=yf
            outputs[f'r{rank}_{stream}_theta']=theta
            er.append(E);res.append(residual)
            # Read-only Gaussian variance response diagnostics of the same active error.
            wa=W[I];d=np.exp(-a['mu']**2/(2*a['v']))/np.sqrt(2*np.pi)/(2*np.sqrt(a['v']))
            evar=d*np.sum(wa*(E@wa),axis=0);rvar=d*np.sum(wa*(residual@wa),axis=0)
            varerr.append(evar);varres.append(rvar);truth_means[rank].append(truefinal)
            stream_rows.append(dict(stream=stream,truth=pin(path),affine_mean_relative=mr,affine_covariance_relative=cr,
                projection_residual_orthogonality=residual_orth,covariance_residual_ratio=float(np.sum(residual*residual)/np.sum(E*E)),
                variance_response_residual_ratio=float(np.sum(rvar*rvar)/np.sum(evar*evar))))
            del trueC,preC,changed,full
        base_energy=energy(er);res_energy=energy(res);vb=energy(varerr);vr=energy(varres)
        row=dict(rank=rank,basis_rank=Q.shape[1],basis_orthogonality=orth,active_count=len(I),cold_count=int(cold.sum()),
            zero_maxabs=zm,zero_rms=zrm,streams=stream_rows,covariance_target=base_energy,covariance_residual=res_energy,
            variance_response_target=vb,variance_response_residual=vr)
        if rank==35:
            parent=outputs['r35_parent'].astype(np.float64)
            baseline=energy([t-parent for t in truth_means[rank]])
            row['cross_final']=dict(parent=baseline)
            for arm in ['projected','full']:
                vals=[t-outputs[f'r35_{j}_{arm}'].astype(np.float64) for j,t in enumerate(truth_means[rank])]
                ev=energy(vals);ev['cross_ratio']=ev['cross_mse']/baseline['cross_mse'] if baseline['cross_mse']>0 else None
                row['cross_final'][arm]=ev
        records.append(row);print(__import__('json').dumps(dict(rank=rank,basis_rank=Q.shape[1],streams=stream_rows,cross_final=row.get('cross_final'))),flush=True)
        del a,C,W,Q,er,res;gc.collect()
    assert len(replay_rows)==9
    predictions=HERE/'fixed_projections_predictions.npz';np.savez(predictions,**outputs)
    write(HERE/'before_1B_truth.json',dict(predictions=pin(predictions),records=records,replays=replay_rows,all_predictions_fixed=True,ordinary_1B_means_accessed=False))
    for row in records:
        rank=row['rank'];path=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
        with np.load(path,allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        parent=outputs[f'r{rank}_parent'].astype(np.float64);base=float(np.mean((truth-parent)**2));sc=[]
        expected={35:1.8706028490077942e-8,100:1.8424534175274928e-8}[rank];assert base==expected,(base,expected)
        for j in range(len(row['streams'])):
            r=dict(stream=j)
            for arm in ['projected','full']:
                y=outputs[f'r{rank}_{j}_{arm}'].astype(np.float64);mse=float(np.mean((truth-y)**2))
                r[arm]=dict(mse=mse,ratio=mse/base)
            sc.append(r)
        row['ordinary_1B']=dict(reference=pin(path),parent_mse=base,streams=sc)
        print(__import__('json').dumps(dict(rank=rank,scores=sc)),flush=True)
    cross=records[0]['cross_final'];cross_pass=cross['parent']['cross_mse']>0 and cross['projected']['cross_ratio']<=.75
    ordinary_pass=all(s['projected']['ratio']<=.80 for r in records for s in r['ordinary_1B']['streams'])
    result=dict(completed=True,passed=bool(cross_pass and ordinary_pass),cross_gate=bool(cross_pass),ordinary_gate=bool(ordinary_pass),
        records=records,replays=replay_rows,setup_flops=setup.summary_dict()['flops_used'],full_estimator_calls=0,native_final_replays=9,
        fitted_parameters=0,new_samples=0,predictions=pin(predictions),wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        qualification='Privileged covariance-space projections on reused development networks. No deployable supplier, complete bill, adjusted score or population guarantee.')
    write(HERE/'result.json',result);print(__import__('json').dumps(dict(completed=True,passed=result['passed'],wall_s=result['wall_s'],peak_rss_bytes=result['peak_rss_bytes'])),flush=True)
if __name__=='__main__':main()
