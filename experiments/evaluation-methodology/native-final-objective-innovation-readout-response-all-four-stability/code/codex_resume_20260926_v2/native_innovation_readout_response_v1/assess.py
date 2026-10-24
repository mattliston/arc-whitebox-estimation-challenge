from common import *
verify();guard('prepare');guard('zero');rows=[];h=.125
for direction in (0,7):
    byamp={}
    for amp in (h,-h,h/2,-h/2):
        name=mode_name(direction,amp);guard(name);r=read(HERE/name/'complete.json')
        assert r['only_output_changed'] and not r['final_targets_read']
        byamp[amp]={x['mini']:packet(x['packet']) for x in r['rows']}
    zero={x['mini']:packet(x['packet'])['prediction'][-1] for x in read(HERE/'zero/complete.json')['rows']}
    for mini in MINIS:
        p,m,ph,mh=[byamp[amp][mini]['prediction'][-1] for amp in (h,-h,h/2,-h/2)]
        rh=(p-m)/(2*h);rl=(ph-mh)/h
        discrepancy=float(np.linalg.norm(rh-rl)/max(np.linalg.norm(rl),1e-30))
        rms=float(np.sqrt(np.mean((h*rl)**2)))
        curvature=float(np.linalg.norm(p+m-2*zero[mini])/max(np.linalg.norm(p-m),1e-30))
        # Independent scalar sums reproduce the vector diagnostics.
        squared=lambda x:math.fsum(float(v)**2 for v in x)
        rd=math.sqrt(squared(rh-rl)/max(squared(rl),1e-60))
        rr=math.sqrt(squared(h*rl)/1024)
        rc=math.sqrt(squared(p+m-2*zero[mini])/max(squared(p-m),1e-60))
        assert all(math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-15) for a,b in ((discrepancy,rd),(rms,rr),(curvature,rc)))
        gates=dict(secant=discrepancy<=.05,resolvable=rms>=5e-7,curvature=curvature<=.10)
        rows.append(dict(direction=direction,mini=mini,secant_relative=discrepancy,response_rms=rms,curvature=curvature,gates=gates,passed=all(gates.values())))
passed=all(x['passed'] for x in rows)
once(HERE/'result.json',dict(passed=passed,rows=rows,basis=read(HERE/'basis_frozen.json'),
    final_targets_read=False,probe_accuracy_scored=False,coefficients_fitted=False,
    prior_failures_unchanged=True,verification_passed=True,limited_to_two_directions_two_networks=True))
print(json.dumps(dict(passed=passed,rows=rows)),flush=True)
