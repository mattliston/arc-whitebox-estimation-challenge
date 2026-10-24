"""Fixed75%-pair v-leg contraction with an exact row-mean control variate.

One deployable ranking: centered Dv column energy times Fv/2 frame-column
energy. A full physical-output energy ranking is diagnostic only. Both use
the same frozen keep fraction, rounded up64; no truth/endpoint fitting.
Every selected contracted column represents a COMPLETE signed +/- pair.
The separate common Su/Fu contribution remains unchanged and is not captured.
"""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
import numpy as np

SOURCE='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def norm2(x):return float(np.sum(np.square(x,dtype=np.float64),dtype=np.float64))

def records(capture,rank):
    receipts=sorted(capture.glob('part*/old_dv_capture_v*/CAPTURE_COMPLETE_v1.json'))
    if rank==35:receipts+=sorted(capture.glob('part*/old_dv_capture_v1/SALVAGED_CAPTURE_v1.json'))
    assert receipts;found=[];pins=[]
    for receipt in receipts:
        done=json.loads(receipt.read_text())
        if rank not in done['ranks']:continue
        if receipt.name=='SALVAGED_CAPTURE_v1.json':
            assert done['status']=='SALVAGED_DIAGNOSTIC_ONLY_FROM_POST_PREDICTION_BOOKKEEPING_FAILURE'
            assert done['diagnostic_only'] and not done['original_capture_successful'] and not done['original_guard_successful']
            assert done['resource_checks_passed'] and done['all_prediction_bytes_equal'] and done['traceback_only_wrong_finish_layer_assert']
        else:assert done['complete']
        assert done['target_free'] and done['source_sha256']==SOURCE
        pins.append(dict(path=str(receipt),sha256=sha(receipt),salvaged_diagnostic_only=receipt.name=='SALVAGED_CAPTURE_v1.json'))
        for entry in done['files']:
            path=receipt.parent/entry['path'];assert sha(path)==entry['sha256'];rec=json.loads(path.read_text())
            if rec['rank']==rank:found.append((path,rec))
    assert {rec['layer']for _,rec in found}==set(range(10,15))
    return found,pins

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank35',type=Path,required=True);ap.add_argument('--rank150',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert os.environ.get('ARC_BLAS_THREADS')=='1'
    a.out.mkdir(parents=True,exist_ok=False);begin=time.monotonic();reports=[];receipts=[]
    for rank,capture in ((35,a.rank35),(150,a.rank150)):
        rows,pins=records(capture,rank);receipts+=pins
        for path,rec in rows:
            def read(key):
                r=rec[key];p=path.parent/r['path'];assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'];x=np.fromfile(p,dtype=np.dtype(r['dtype'])).reshape(r['shape']);assert np.isfinite(x).all();return x
            D=read('Dv').astype('d');F=read('FvT').astype('d');native=read('native_inner').astype('d');U=read('pre_basis').astype('d');live=read('live_indices').astype('i8');R=U[live].T
            if 'lazy_map'in rec:R=read('lazy_map').astype('d').T@R
            m,P=D.shape;r=F.shape[1]
            assert F.shape[0]==P and native.shape==(m,r) and R.shape[0]==r
            assert sum(int(h)for _,h in rec['pair_blocks'])==P
            I=D@F;physical=I@R;den=max(norm2(physical),1e-300);iden=max(norm2(I),1e-300)
            mean=D.mean(axis=0,keepdims=True);centered=D-mean;exact_mean=mean@F
            denergy=np.sum(centered*centered,axis=0);frameenergy=np.sum(F*F,axis=1)
            keep=min(P,((3*P+3)//4+63)//64*64)
            index=np.sort(np.argsort(-(denergy*frameenergy),kind='stable')[:keep]);drop=np.setdiff1d(np.arange(P),index,assume_unique=True)
            # Omitted residual has every signed-energy cross term: compute the
            # complete dropped contribution, then map to original physical rows.
            omitted=centered[:,drop]@F[drop]
            ordinary_omitted=D[:,drop]@F[drop]
            def error(delta):
                ph=delta@R
                return dict(inner_relative_l2=float(np.sqrt(norm2(delta)/iden)),physical_v_leg_relative_l2=float(np.sqrt(norm2(ph)/den)),physical_max_abs=float(np.max(np.abs(ph))),physical_signed_mean=float(ph.mean()),physical_denominator_norm2=den)
            fphysical=F@R
            oracle_score=denergy*np.sum(fphysical*fphysical,axis=1)
            oracle_index=np.sort(np.argsort(-oracle_score,kind='stable')[:keep]);oracle_drop=np.setdiff1d(np.arange(P),oracle_index,assume_unique=True)
            oracle_omitted=centered[:,oracle_drop]@F[oracle_drop]
            result=dict(rank=rank,layer=rec['layer'],part=rec['part'],record_path=str(path),record_sha256=sha(path),shape=dict(Dv=list(D.shape),FvT=list(F.shape),physical_R=list(R.shape)),pair_blocks=rec['pair_blocks'],complete_pair_columns=True,keep=keep,drop=P-keep,fraction_kept=keep/P,
                original_product_f32_rounding=error(native-I),fixed_centered_frame_score=error(omitted),same_selection_without_mean_rescue=error(ordinary_omitted),diagnostic_physical_score_oracle=error(oracle_omitted),
                centered_v_leg_fraction=norm2((I-exact_mean)@R)/den,mean_only_error=error(I-exact_mean),
                independent_column_proxy_omitted_fraction=float(np.sum((denergy*frameenergy)[drop])/max(np.sum(denergy*frameenergy),1e-300)),
                preserved='All384 v-frame coordinates, all physical output rows, entire common Su/Fu term and exact fullv rowmean. Full carried bank/diagonal/NPforms remain parent. Only selected completepair v-residual contributions omitted.',
                candidate_index_sha256=hashlib.sha256(index.astype('<i8').tobytes()).hexdigest(),score_gain_claim=False)
            reports.append(result);out=a.out/f'rank{rank:04d}_L{rec["layer"]:02d}_{rec["part"]}_RESULT_v1.json'
            with out.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
            print(json.dumps(dict(rank=rank,layer=rec['layer'],part=rec['part'],keep=keep,P=P,candidate_physical_v_l2=result['fixed_centered_frame_score']['physical_v_leg_relative_l2'],plain_same_indices_l2=result['same_selection_without_mean_rescue']['physical_v_leg_relative_l2'],diagnostic_physical_score_l2=result['diagnostic_physical_score_oracle']['physical_v_leg_relative_l2'])),flush=True)
            del D,F,native,U,live,R,I,physical,mean,centered,exact_mean,denergy,frameenergy,omitted,ordinary_omitted,fphysical,oracle_score,oracle_omitted
    receipt=dict(utc=utc(),status='TARGET_FREE_FIXED_PAIR_CONTRACTED_DIMENSION_PHYSICAL_DIAGNOSTIC',source_sha256=SOURCE,script_sha256=sha(__file__),capture_pins=receipts,records=reports,train_ranks=[35,150],truth_access=False,heldout_access=False,native_launches=0,fit=False,keep_fraction=.75,roundup=64,main_score='centeredDv columnL2 × Fv2 framecolumnL2',oracle_only='Exact physicaloutput columnL2; not deployable withoutadditionalpricing',not_full_old_read_denominator=True,elapsed_s=time.monotonic()-begin,peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with(a.out/'RESULT_v1.json').open('x')as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
