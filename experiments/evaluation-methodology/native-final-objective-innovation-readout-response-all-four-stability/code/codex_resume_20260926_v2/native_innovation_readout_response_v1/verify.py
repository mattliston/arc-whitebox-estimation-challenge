"""Independent immutable readout, source, prefix and response-result audit."""
from common import *
verify();guard('assessment');result=read(HERE/'result.json');assert not result['passed']
assert not result['final_targets_read'] and not result['probe_accuracy_scored'] and not result['coefficients_fitted']
basis=packet(result['basis']['packet']);G=basis['gram'];V=basis['directions'];e=basis['eigenvalues'];scale=float(basis['source_scale'])
assert G.shape==(32,32) and V.shape==(32,8) and np.all(np.diff(e)<=0)
np.testing.assert_allclose(G,G.T,rtol=1e-13,atol=1e-20)
np.testing.assert_allclose(V.T@G@V,scale*np.eye(8),rtol=1e-8,atol=scale*1e-9)
for j in range(8):
    assert V[np.argmax(abs(V[:,j])),j]>0
    np.testing.assert_allclose(G@V[:,j],e[j]*V[:,j],rtol=1e-7,atol=1e-14)
base={x['mini']:x for x in read(BASE/'capture/complete.json')['rows']}
records=[]
names=['zero']+[mode_name(d,a) for d in (0,7) for a in (.125,-.125,.0625,-.0625)]
for name in names:
    guard(name);r=read(HERE/name/'complete.json')
    assert not r['final_targets_read'] and r['only_output_changed'] and r['output_bias_zero']
    expected=np.zeros((1,32),dtype='float32') if name=='zero' else (r['amplitude']*V[:,r['direction']]).astype('float32')[None,:]
    for row in r['rows']:
        a=packet(row['packet']);parent=packet(base[row['mini']]['packet'],['prediction'])['prediction']
        assert a['readout'].tobytes()==expected.tobytes()
        prefix=16 if name=='zero' else 9
        assert a['prediction'][:prefix].tobytes()==parent[:prefix].tobytes()
        assert np.isfinite(a['prediction']).all() and np.min(a['prediction'])>=0
        records.append(dict(name=name,mini=row['mini'],readout_and_prefix_exact=True))
assert len(records)==18
assert all(not r['gates']['secant'] and not r['gates']['curvature'] and r['gates']['resolvable'] for r in result['rows'])
once(HERE/'verification.json',dict(passed=True,rows=records,basis_identities_passed=True,
    zero_all16_exact=True,probe_first9_exact=True,all_readouts_exact=True,
    final_targets_not_used=True,result=pin(HERE/'result.json')))
print('READOUT_AND_SOURCE_VERIFICATION_PASSED',flush=True)
