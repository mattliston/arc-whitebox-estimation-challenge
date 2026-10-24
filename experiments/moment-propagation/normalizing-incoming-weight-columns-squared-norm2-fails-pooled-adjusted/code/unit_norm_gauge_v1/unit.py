"""Independent direct-network certificate for the paid coordinate transform."""
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((HERE/'build.json').read_text())
assert sha(HERE/'candidate.py')==build['candidate_sha256']
assert sha(HERE/'gauge_fnp.py')==build['fragment_sha256']
assert all(t['num_threads']==1 for t in threadpool_info())
namespace = {'fnp':fnp}
exec(compile((HERE/'gauge_fnp.py').read_text(),str(HERE/'gauge_fnp.py'),'exec'),namespace)
normalize = namespace['_unitnorm_weights']
rng = np.random.default_rng(2026092006)
base = (rng.normal(size=(5,32,32))*np.sqrt(2/32)).astype(np.float32)
spread = (base*np.exp(rng.uniform(-1,1,size=(5,1,32)))).astype(np.float32)
zero = base.copy()
zero[0,:,3] = 0
zero[2,:,7] = 0
zero[4,:,2] = 0
fixtures = [('he',base),('unequal_columns',spread),('zero_columns',zero)]
inputs = rng.normal(size=(17,32))


def run():
    arrays,rows,rejections = [],[],[]
    with flops.BudgetContext(flop_budget=10**8,quiet=True) as ctx:
        for name,original in fixtures:
            weights = [fnp.asarray(w.copy()) for w in original]
            before = [np.asarray(w).tobytes() for w in weights]
            ww,ss = normalize(weights)
            out = np.stack([np.asarray(w).copy() for w in ww])
            scales = np.asarray(ss).copy()
            assert out.dtype==np.float32 and scales.dtype==np.float64
            assert out.shape==original.shape and scales.shape==(5,32)
            assert np.isfinite(out).all() and np.isfinite(scales).all() and (scales>0).all()
            assert np.array_equal(scales[-1],np.ones(32))
            assert before==[np.asarray(w).tobytes() for w in weights]
            a,b = inputs.copy(),inputs.copy()
            previous = np.ones(32)
            reconstruction,forward,norm_errors = [],[],[]
            for layer in range(5):
                restored_weight = out[layer].astype(np.float64)*previous[:,None]/scales[layer][None,:]
                reconstruction.append(float(np.max(np.abs(restored_weight-original[layer]))/max(1.,float(np.max(np.abs(original[layer]))))))
                a = np.maximum(a@original[layer].astype(np.float64),0)
                b = np.maximum(b@out[layer].astype(np.float64),0)
                restored = b/scales[layer]
                forward.append(float(np.max(np.abs(a-restored))/max(1.,float(np.max(np.abs(a))))))
                energy = np.sum(out[layer].astype(np.float64)**2,axis=0)
                nonzero = np.sum(original[layer].astype(np.float64)**2,axis=0)>0
                if layer<4:
                    norm_errors.append(float(np.max(np.abs(energy[nonzero]-2))))
                    assert np.array_equal(energy[~nonzero],np.zeros(np.count_nonzero(~nonzero)))
                previous = scales[layer]
            assert max(reconstruction)<=2e-7,(name,reconstruction)
            assert max(forward)<=3e-6,(name,forward)
            assert max(norm_errors)<=2e-6,(name,norm_errors)
            rows.append(dict(case=name,weight_reconstruction_scaled_errors=reconstruction,
                             layerwise_direct_output_scaled_errors=forward,
                             column_squared_norm_errors=norm_errors,
                             final_scale_exactly_one=True,input_bytes_unchanged=True))
            arrays.extend([out,scales])
        bad = [('short_list',[fnp.ones((3,3))]),
               ('nonsquare',[fnp.ones((3,2)),fnp.ones((3,3))]),
               ('mismatched_shape',[fnp.ones((3,3)),fnp.ones((4,4))]),
               ('nonfinite',[fnp.full((3,3),float('nan')),fnp.ones((3,3))])]
        for name,ww in bad:
            try:
                normalize(ww)
            except RuntimeError:
                rejections.append(name)
            else:
                raise AssertionError(name)
    return arrays,rows,rejections,ctx.summary_dict()['flops_used']


arrays,rows,rejections,bill = run()
again,rows2,rejections2,bill2 = run()
assert rows==rows2 and rejections==rejections2 and bill==bill2
assert all(a.tobytes()==b.tobytes() for a,b in zip(arrays,again))
out = HERE/'unit_v1'
out.mkdir(exist_ok=False)
record = dict(passed=True,cases=rows,rejections=rejections,complete_bill_per_batch=bill,
              repeated_arrays_byte_exact=True,repeated_bills_exact=True,
              source_sha256=sha(__file__),candidate_sha256=build['candidate_sha256'],
              target_arrays_opened=False,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
