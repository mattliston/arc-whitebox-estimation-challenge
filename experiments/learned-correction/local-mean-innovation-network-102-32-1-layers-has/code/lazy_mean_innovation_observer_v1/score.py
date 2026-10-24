import json,time,resource
from common import HERE,CHECK,read,write,pin,sha,verify
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify('learning_sources_sealed.json');seal=verify('forecasts_sealed.json');assert seal['all8_predictions_fixed'] and not seal['check_mean_arrays_accessed']
    g=read(HERE/'predict_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    assert read(HERE/'fit_result.json')['passed'];meta=read(HERE/'learning_inputs.json');started=time.monotonic();rows=[]
    for rank in CHECK:
        rec=[r for r in seal['records'] if r['rank']==rank][0];tp=meta['check_truth'][str(rank)]
        assert sha(tp['path'])==tp['sha256']
        with np.load(tp['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        with np.load(rec['forecast']['path'],allow_pickle=False) as z:prediction=z['prediction'][-1].astype(np.float64)
        parent=np.load(rec['parent_prediction']['path'],allow_pickle=False)[-1].astype(np.float64)
        pmse=float(np.mean((parent-truth)**2));mse=float(np.mean((prediction-truth)**2))
        assert pmse==meta['expected_parent_raw'][str(rank)]
        rows.append(dict(rank=rank,parent_raw=pmse,candidate_raw=mse,raw_ratio=mse/pmse,passed=mse/pmse<=.85))
    result=dict(completed=True,passed=all(r['passed'] for r in rows),gate='both CHECK ordinary1B raw ratios <=.85',rows=rows,
        pooled_raw_ratio=sum(r['candidate_raw'] for r in rows)/sum(r['parent_raw'] for r in rows),
        network_disjoint_fit=True,pristine_validation=False,adjusted_score=False,production_cost_unmeasured=True,
        forecasts_seal=pin(HERE/'forecasts_sealed.json'),new_samples=0,wall_s=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'score.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
