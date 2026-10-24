import gc,json,resource,time
from common import *
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();seal=verify('captures_sealed.json')
    rows=[];start=time.monotonic()
    for row in seal['rows']:
        with np.load(row['capture']['path'],allow_pickle=False) as z:
            t={k[6:]:z[k] for k in z.files if k.startswith('param_')}
            for l in range(8,15):
                d={k:z[f'edge_{k}_{l:02d}'] for k in ('F','R','S','pf','gf','sf')};B=basis(d,t)
                reference=z[f'edge_correction_{l:02d}'];q=B@t['output'].astype(np.float64)
                rel=float(np.linalg.norm(q-reference)/max(np.linalg.norm(reference),1e-30))
                assert rel<=3e-6,(row['rank'],l,rel)
                assert np.isfinite(B).all() and np.array_equal(B,B.transpose(1,0,2))
                rows.append(dict(rank=row['rank'],layer=l,base_reconstruction_relative=rel))
                del B,d,q,reference;gc.collect()
        print(json.dumps(dict(rank=row['rank'],max_relative=max(x['base_reconstruction_relative'] for x in rows if x['rank']==row['rank']))),flush=True)
    write(HERE/'wide_qualification.json',dict(passed=True,rows=rows,targets_accessed=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':main()
