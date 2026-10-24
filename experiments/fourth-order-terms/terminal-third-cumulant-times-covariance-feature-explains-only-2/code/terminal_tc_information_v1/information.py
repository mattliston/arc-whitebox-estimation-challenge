"""Fixed optimistic terminal TC scalar test; feature seal precedes targets."""
import hashlib,importlib.util,json,resource,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
from algebra import source

H=Path(__file__).resolve().parent;D=H.parent;P2=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
seal=read(H/'source_sealed.json')
for name,digest in seal['files_sha256'].items():assert sha(H/name)==digest,name
for path,digest in seal['external_sha256'].items():assert sha(path)==digest,path
guard=read(H/'capture_watch.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
capt=read(H/'capture/completed.json');assert capt['completed'] and capt['all16_and_bills_exact'] and capt['repeat_packet_arrays_exact']
assert read(H/'unit.json')['passed']
assert all(x['num_threads']==1 for x in threadpool_info())
oracle_path=D/'shared_fourth_odd_lineage_v1/oracle.py'
spec=importlib.util.spec_from_file_location('fixed_scalar_oracle',oracle_path);oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
out=H/'information';out.mkdir(exist_ok=False);features=[];first=None;t0=time.monotonic()
for rec in capt['records']:
    assert sha(rec['packet'])==rec['packet_sha256']
    with np.load(rec['packet'],allow_pickle=False) as z:packet={k:z[k] for k in z.files}
    for k,meta in rec['arrays'].items():
        a=packet[k];assert list(a.shape)==meta['shape'] and str(a.dtype)==meta['dtype']
        assert hashlib.sha256(a.tobytes()).hexdigest()==meta['raw_sha256']
    arrays,report=source(packet);arrays['prediction']=packet['prediction']
    if rec['call']==0:first={k:v.copy() for k,v in arrays.items()}
    if rec['call']==2:assert all(v.tobytes()==first[k].tobytes() for k,v in arrays.items())
    target=out/f"feature_{rec['call']:02d}.npz";np.savez(target,**arrays)
    row=dict(call=rec['call'],rank=rec['rank'],path=str(target),sha256=sha(target),packet_sha256=rec['packet_sha256'],parent_flops=rec['flops'],report=report)
    features.append(row);print(json.dumps(row),flush=True)
    del packet,arrays
pretruth=dict(features=features,capture_receipt_sha256=sha(H/'capture/completed.json'),repeat_feature_bytes_exact=True,
              source_seal_sha256=sha(H/'source_sealed.json'),truth_accessed=False)
with (out/'features_sealed_before_targets.json').open('x') as f:json.dump(pretruth,f,indent=2)
target_hashes={35:'ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61',100:'637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c'}
parent_raw={35:1.8706028490077942e-8,100:1.8424534175274928e-8}
results=[]
for row in features[:2]:
    rank=row['rank'];target=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
    assert sha(target)==target_hashes[rank]
    with np.load(target,allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
    assert sha(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as z:p=z['prediction'][-1].astype(np.float64);feature=z['feature'].astype(np.float64)
    raw=float(np.mean((p-truth)**2));assert np.isclose(raw,parent_raw[rank],rtol=2e-13,atol=0),(raw,parent_raw[rank])
    fit,ideal,served=oracle.clipped_scalar_oracle(p,feature,truth)
    ratios={k:fit[k]/raw for k in ['ideal_raw','served_raw','rounding_allowance_lower_raw']}
    fixed_raw=float(np.mean((np.maximum(p+feature,0)-truth)**2))
    result=dict(rank=rank,parent_raw=raw,truth_sha256=sha(target),oracle=fit,ratios=ratios,
                coefficient_one_raw_ratio=fixed_raw/raw,
                gate_passed=ratios['ideal_raw']<=.80 and ratios['served_raw']<=.80,
                target_conditioned=True,incremental_cost_optimistically_zero=True)
    results.append(result);print(json.dumps(result),flush=True)
record=dict(completed=True,records=results,information_gate_passed=all(r['gate_passed'] for r in results),
            threshold=.80,features_seal_sha256=sha(out/'features_sealed_before_targets.json'),
            feature_repeat_byte_exact=True,no_deployable_fit=True,no_candidate=True,
            actual_adjusted_candidate_score_computed=False,elapsed_s=time.monotonic()-t0,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({'completed':True,'information_gate_passed':record['information_gate_passed']}),flush=True)
