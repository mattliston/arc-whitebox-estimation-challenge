"""Fixed common-u75 diagnostic with aligned full-old-read physical denominator.

TRAIN35/150 only, no truth, fitting, or keep-fraction search. The v75 error is
also rebased on full u+v where exact capture geometry and prediction hashes
agree. Rank8 residual oracle is diagnostic only, not a deployable algorithm.
"""
import argparse,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'old_vpair_prune_v1'))
from analyze_v1 import records as vrecords,norm2,sha,SOURCE,utc

def read(path,rec,key):
    r=rec[key];p=path.parent/r['path'];assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'];x=np.fromfile(p,dtype=r['dtype']).reshape(r['shape']);assert np.isfinite(x).all();return x

def selected(D,F):
    m,P=D.shape;mean=np.mean(D,axis=0,dtype=np.float32);keep=min(P,(((3*P+3)//4)+63)//64*64)
    score=np.maximum(np.sum(D*D,axis=0,dtype=np.float64)-m*mean.astype('d')**2,0.)*np.sum(F.T*F.T,axis=0,dtype=np.float64)
    return np.sort(np.argsort(-score,kind='stable')[:keep])

def delta(D,F,index):
    mean=np.mean(D,axis=0,dtype=np.float32)
    full=D.astype('d')@F.astype('d')
    compact=(D[:,index]-mean[index]).astype('d')@F[index].astype('d')+mean.astype('d')[None,:]@F.astype('d')
    return compact-full,full

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--rank35',type=Path,required=True);p.add_argument('--rank150',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('ARC_BLAS_THREADS')=='1';a.out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    complete=json.loads((a.capture/'COMPLETE.json').read_text());assert complete['successful']and not complete['pending']and all(v==0 for v in complete['results'].values())
    controls={}
    for path in a.control.glob('part*/rows.jsonl'):
        manifest=json.loads((path.parent/'manifest.json').read_text());assert manifest['sha256']==SOURCE
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['rank']in(35,150)and row['phase']=='scored':
                pred=path.parent/('pred_'+str(row['rank'])+'.bin');assert sha(pred)==row['prediction_sha256'];controls[row['rank']]=row['prediction_sha256']
    assert set(controls)=={35,150}
    urecs={};parity=[];guards=[]
    for part in sorted(a.capture.glob('part[0-9][0-9]')):
        guard=a.capture/(part.name+'_GUARD.json');g=json.loads(guard.read_text());assert g['successful']and g['whole_group_retired']and g['child_exit']==0 and g['max_owned_swap_bytes']==0
        launch=json.loads((part/'launcher.json').read_text());assert launch['client_exit']==0 and launch['thread_provenance']['blas_threads']==4
        manifest=json.loads((part/'manifest.json').read_text());assert manifest['sha256']==SOURCE and manifest['diagnostic_only']
        rows=[json.loads(s)for s in(part/'diagnostic_rows.jsonl').read_text().splitlines()];assert len(rows)==1;row=rows[0];rank=row['rank'];assert row['phase']=='scored'and rank in(35,150)
        pred=part/('diagnostic_pred_'+str(rank)+'.bin');assert sha(pred)==row['prediction_sha256']==controls[rank]
        receipt=part/'old_su_capture_v1/CAPTURE_COMPLETE_v1.json';r=json.loads(receipt.read_text());assert r['complete']and r['ranks']==[rank]and r['source_sha256']==SOURCE and r['target_free']and r['native_handles_retained']is False
        for entry in r['files']:
            path=receipt.parent/entry['path'];assert sha(path)==entry['sha256'];rec=json.loads(path.read_text());urecs[(rec['rank'],rec['layer'],rec['part'])]=(path,rec)
        parity.append(dict(rank=rank,prediction_sha256=row['prediction_sha256'],exact_control_bytes=True,capture_receipt_sha256=sha(receipt)))
        guards.append(dict(path=str(guard),sha256=sha(guard),peak_rss=g['peak_group_rss_bytes'],zero_swap=True))
    assert len(urecs)==14
    reports=[];vpins=[]
    for rank,vcapture in((35,a.rank35),(150,a.rank150)):
        vrows,pins=vrecords(vcapture,rank);vpins+=pins;ui=vi=None;uf=vf=None
        # Prior capture prediction bytes must also equal the same exact control.
        oldrows=vcapture/'part01/diagnostic_rows.jsonl';oldrow=[json.loads(x)for x in oldrows.read_text().splitlines()if json.loads(x)['rank']==rank][0];assert oldrow['prediction_sha256']==controls[rank]
        for vp,vr in sorted(vrows,key=lambda x:(x[1]['layer'],x[1]['part'])):
            if vr['part']=='join':continue
            key=(rank,vr['layer'],vr['part']);up,ur=urecs[key]
            for name in('pre_basis','live_indices'):
                assert ur[name]['sha256']==vr[name]['sha256']and ur[name]['shape']==vr[name]['shape']
            assert ('lazy_map'in ur)==('lazy_map'in vr)
            if 'lazy_map'in ur:assert ur['lazy_map']['sha256']==vr['lazy_map']['sha256']
            assert ur['pair_blocks']==vr['pair_blocks']and ur['common_rank']==vr['common_rank']
            Su=read(up,ur,'Su');Fu=read(up,ur,'FuT');Un=read(up,ur,'native_inner');Dv=read(vp,vr,'Dv');Fv=read(vp,vr,'FvT');Vn=read(vp,vr,'native_inner')
            R=read(up,ur,'pre_basis').astype('d')[read(up,ur,'live_indices').astype('i8')].T
            if 'lazy_map'in ur:R=read(up,ur,'lazy_map').astype('d').T@R
            ru=ur['common_rank'];assert Fu.shape[1]==ru and Un.shape==(Su.shape[0],ru)and Vn.shape[0]==Su.shape[0]
            if ur['layer']==10:
                ui=selected(Su,Fu);vi=selected(Dv,Fv);uf=Fu.copy();vf=Fv.copy()
            assert ui is not None and np.array_equal(uf,Fu)and np.array_equal(vf,Fv)
            du,fullu=delta(Su,Fu,ui);dv,fullv=delta(Dv,Fv,vi);eu=du@R[:ru];ev=dv@R
            original=Vn.copy();original[:,:ru]+=Un;physical=original.astype('d')@R;den=max(norm2(physical),1e-300)
            uphysical=fullu@R[:ru];vphysical=fullv@R
            # Oracle only: best physical residual rank8 bounds any cheap linear
            # correction, without claiming those singularvectors are available.
            sv=np.linalg.svd(ev,compute_uv=False);rank8=float(np.sum(sv[:8]**2)/max(np.sum(sv**2),1e-300))
            result=dict(rank=rank,layer=ur['layer'],part=ur['part'],original_pairs=Su.shape[1],retained_u_pairs=len(ui),retained_v_pairs=len(vi),fixed_factor_bytes=True,geometry_hashes_identical=True,
                common_u_relative_physical_l2=float(np.sqrt(norm2(eu)/max(norm2(uphysical),1e-300))),u75_relative_full_old_l2=float(np.sqrt(norm2(eu)/den)),
                v75_relative_full_old_l2=float(np.sqrt(norm2(ev)/den)),v75_relative_v_leg_l2=float(np.sqrt(norm2(ev)/max(norm2(vphysical),1e-300))),
                common_u_to_full_norm_ratio=float(np.sqrt(norm2(uphysical)/den)),v_to_full_norm_ratio=float(np.sqrt(norm2(vphysical)/den)),full_physical_norm2=den,
                v_error_rank8_oracle_energy_fraction=rank8,u_index_sha256=hashlib.sha256(ui.astype('<i8').tobytes()).hexdigest(),v_index_sha256=hashlib.sha256(vi.astype('<i8').tobytes()).hexdigest())
            reports.append(result);print(json.dumps(result),flush=True)
    assert len(reports)==10
    result=dict(utc=utc(),successful=True,source_sha256=SOURCE,script_sha256=sha(__file__),capture_complete_sha256=sha(a.capture/'COMPLETE.json'),prediction_parity=parity,guards=guards,prior_v_receipts=vpins,records=reports,
        truth_access=False,heldout_access=False,fit=False,keep_fraction=.75,roundup=64,output_axes_preserved=256,full_mean_preserved_algebraically=True,
        caveat='Physical read diagnostic only. No later cumulant/activation sensitivities or finalMSE. Vrank8 is an undeployable upperbound, not a new fitted candidate.',elapsed_s=time.monotonic()-begin,peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with(a.out/'RESULT_v1.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
