import numpy as np
from common import HERE, read, write, pin, verify, seal_check
from snapshot import factored_reference, physical

seal_check(); assert read(HERE/'unit.json')['passed']
sealed = read(HERE/'captures_sealed.json')
for p in sealed['rows']:
    verify(p)
packets = {label:read(HERE/f'capture_{label}/result.json')
           for label in ['base','plus','minus','double','base_repeat']}
for r in packets.values():
    for key in ['source','weights','prediction','numpy_source']:
        verify(r[key])
    assert r['all16_original_bytes_exact'] and r['bill_less_copies_exact']
    assert not r['targets_opened'] and r['numerical_cache_retired']
    assert r['original_flops'] == r['flops'] - r['copy_flops']
    for row in r['rows']:
        verify(row['trace'])
base, repeat = packets['base'], packets['base_repeat']
assert base['prediction']['sha256'] == repeat['prediction']['sha256']
assert base['flops'] == repeat['flops'] and base['copy_flops'] == repeat['copy_flops']
for a,b in zip(base['rows'],repeat['rows']):
    with np.load(a['trace']['path'], allow_pickle=False) as za, np.load(b['trace']['path'], allow_pickle=False) as zb:
        assert za.files == zb.files
        assert all(za[k].tobytes() == zb[k].tobytes() for k in za.files)

def norm(x):
    return float(np.linalg.norm(x))

def relative(x, y):
    return norm(x-y)/max(norm(y), 1e-300)

def response(values):
    b,p,m,d = [values[k] for k in ['base','plus','minus','double']]
    center = (p-m)/2
    forward = p-b
    cn, fn = norm(center), norm(forward)
    return dict(centered_norm=cn, forward_norm=fn,
                forward_vs_centered_relative=norm(forward-center)/cn if cn else None,
                double_curvature_relative=norm(d-2*p+b)/fn if fn else None)

rows=[]; responses=[]; summary_by_label={label:[] for label in ['base','plus','minus','double']}
for layer in range(5,15):
    native_p={}; ref_p={}; native_y={}; ref_y={}
    for label in ['base','plus','minus','double']:
        source = packets[label]['rows'][layer-5]['trace']
        with np.load(source['path'], allow_pickle=False) as data:
            fields = {k:data[k].copy() for k in data.files}
        y,q,r = [fields[k].astype(np.float64) for k in ['Y','Q','R']]
        q64,r64 = np.linalg.qr(y, mode='reduced')
        assert fields['Q'].tobytes() == q64.astype(np.float32).tobytes()
        assert fields['R'].tobytes() == r64.astype(np.float32).tobytes()
        yr = factored_reference(fields, np)
        qr,rr = np.linalg.qr(yr, mode='reduced')
        singular = np.linalg.svd(r, compute_uv=False)
        singular_ref = np.linalg.svd(rr, compute_uv=False)
        assert singular[-1] > 0 and singular_ref[-1] > 0
        previous=fields['previous'].astype(np.int64)
        qp = physical(q, previous, np); qrp = physical(qr, previous, np)
        native_p[label] = qp @ qp.T
        ref_p[label] = qrp @ qrp.T
        native_y[label] = physical(y, previous, np)
        ref_y[label] = physical(yr, previous, np)
        sketch_delta=norm(y-yr)
        row=dict(layer=layer,label=label,rows=len(previous),rank=384,
            native_y_norm=norm(y),reference_y_norm=norm(yr),
            native_condition=float(singular[0]/singular[-1]),
            reference_condition=float(singular_ref[0]/singular_ref[-1]),
            native_singular_values=singular.tolist(),reference_singular_values=singular_ref.tolist(),
            QR_reconstruction_relative=relative(q@r,y),
            Q_orthogonality_relative=norm(q.T@q-np.eye(384))/np.sqrt(384),
            same_native_input_QR64_rounded_bytes_exact=True,
            sketch_rounding_relative=sketch_delta/max(norm(yr),1e-300),
            sketch_delta_frobenius_over_reference_sigma_min=sketch_delta/float(singular_ref[-1]),
            projector_difference_frobenius=norm(native_p[label]-ref_p[label]),
            projector_difference_relative=norm(native_p[label]-ref_p[label])/np.sqrt(384),
            trace=source)
        assert row['QR_reconstruction_relative'] < 1e-5 and row['Q_orthogonality_relative'] < 1e-5
        rows.append(row);summary_by_label[label].append({k:v for k,v in row.items() if 'singular_values' not in k and k!='trace'})
    responses.append(dict(layer=layer,native_sketch=response(native_y),reference_sketch=response(ref_y),
                          native_projector=response(native_p),reference_projector=response(ref_p),
                          base_sketch_rounding_over_centered_response=norm(native_y['base']-ref_y['base'])/
                              max(norm((native_y['plus']-native_y['minus'])/2),1e-300)))
    print(dict(layer=layer,base_condition=rows[-4]['native_condition'],
               base_sketch_rounding=rows[-4]['sketch_rounding_relative'],
               base_projector_difference=rows[-4]['projector_difference_relative'],
               native_projector_response=responses[-1]['native_projector'],
               reference_projector_response=responses[-1]['reference_projector']),flush=True)
    del native_p,ref_p,native_y,ref_y,fields,y,q,r,yr,qr,rr

outputs={label:np.load(packets[label]['prediction']['path'],allow_pickle=False).astype(np.float64)
         for label in ['base','plus','minus','double']}
output_responses=[dict(layer=layer,**response({k:v[layer] for k,v in outputs.items()})) for layer in range(16)]
result=dict(completed=True,capture_qualification_passed=True,repeat_trace_prediction_bill_exact=True,
            actual_QR_widens_confirmed=True,rows=rows,layer_responses=responses,
            output_responses=output_responses,summary_by_label=summary_by_label,
            full_native_calls=5,full_native_flops=sum(x['flops'] for x in packets.values()),
            diagnostic_copy_flops=sum(x['copy_flops'] for x in packets.values()),
            targets_opened=False,fit_steps=0,candidate_created=False,upload=False,
            scope='Frozen-current-input sketch rounding and gauge-invariant response. No causal end-to-end accuracy conclusion.')
write(HERE/'analysis.json',result)
print(dict(completed=True,full_native_calls=5,targets_opened=False),flush=True)
