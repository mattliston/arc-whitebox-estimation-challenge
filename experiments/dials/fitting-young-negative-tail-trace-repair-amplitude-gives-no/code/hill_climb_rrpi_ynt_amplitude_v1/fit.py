"""Root-only one-coordinate secant fit from native BASE and gain0 FIT4."""
from pathlib import Path
import json, math, os, sys
import prepare as b
D=Path(__file__).resolve().parent


def main():
    assert len(sys.argv)==3;s,pp,c=b.configuration(sys.argv[1],sys.argv[2])
    q=b.module(c['engine']['path'],'ynt_amplitude_metrics');f=b.module(c['helper']['path'],'ynt_amplitude_panels');np=q.np
    assert str(Path(sys.executable).absolute())==c['runtime']==str(q.NATIVE) and np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in q.THREADS)
    s.guard(c['direction_arm']['guard'],c['direction_arm']['command'],150)
    truth=f.truths(q,c,'fit4');control=c['controls']['fit4'];assert control['arm']['source']==c['parent']
    base,yy,bp=f.panel(q,c,control['arm'],b.FIT,truth,control['artifacts'])
    plus,zz,zp=f.panel(q,c,c['direction_arm'],b.FIT,truth)
    for rank in b.FIT:assert zz[rank][:7].tobytes()==yy[rank][:7].tobytes()
    direction=[];error=[];per_network=[]
    for row in base:
        rank=row['rank'];scale=math.sqrt(max(.1,row['util'])/1024)
        a=(zz[rank][15]-yy[rank][15])*scale;e=(yy[rank][15]-truth[rank])*scale
        direction.append(a);error.append(e)
        per_network.append(dict(rank=rank,G=float(a@a),g=float(a@e),baseline_weighted_loss=float(e@e)))
    A=np.concatenate(direction);residual=np.concatenate(error);G=float(A@A);g=float(A@residual);Q=1.1*G
    assert all(math.isfinite(v) for v in (G,g,Q)) and G>=0
    x=0. if G==0 else min(2.,max(-2.,-g/Q))
    den=float(residual@residual);pred=float((residual+x*A)@(residual+x*A));assert den>0 and math.isfinite(pred)
    gain=100*(1-pred/den);admitted=gain>=.1 and G>0
    result=dict(passed_integrity=True,endpoint_admitted=admitted,protocol=pp,fitter=q.pin(__file__),parent=c['parent'],
                fit_ranks=b.FIT,alpha=x,gain=1.-x,direction_source=c['direction_source'],G=G,g=g,Q=Q,predicted_parent_weighted_loss_gain_pct=gain,
                forecast_uses_parent_bill_weights=True,forecast_includes_candidate_compute=False,
                actual_candidate_bills_used_at_endpoint_gate=True,secant_step=1.,secant_not_exact_gradient=True,
                nonlinear_even_response_unmeasured=True,per_network=per_network,direction_metrics=q.metrics(base,plus),
                artifacts=dict(base=bp,direction=zp),endpoint=None,check_targets_read=False,
                exact_gradient_claim=False,one_endpoint_only=True,direction_integrity_only_no_score_gate=True,bank_qualified=False)
    if admitted:
        source,proof=b.edit_source(q.checked(c['parent']).decode(),x)
        with (D/'endpoint.py').open('x') as out:out.write(source)
        result.update(endpoint=q.pin(D/'endpoint.py'),source_inverse=proof)
    q.once(D/'fit_result.json',result)
    if admitted:q.once(D/'endpoint_fit4_commands.json',b.commands(s,pp,c,result['endpoint'],'fit4'))
    print(json.dumps({k:result[k] for k in ('endpoint_admitted','alpha','gain','predicted_parent_weighted_loss_gain_pct')}))


if __name__=='__main__':main()
