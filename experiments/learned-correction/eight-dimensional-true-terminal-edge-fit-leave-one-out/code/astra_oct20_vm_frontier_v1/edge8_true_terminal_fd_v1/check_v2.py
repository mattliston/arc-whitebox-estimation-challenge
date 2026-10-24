"""Tiny official proof that all16 literal FD arms are live and cost identical."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import ast,copy,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
NAMES=('_E2Hold','_g5_sub','_g5_chain','_e2_ms','_ps_pre32','_ps_k2132','_ps32',
       '_e2_edge_consts','_message_edge_parameters','_g5_message_edge_base','_ecut_raw','_message_edge_source')
def main():
    assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL')
    build=json.loads((HERE/'BUILD_v1.json').read_text())
    prereg=json.loads((HERE/'PREREG_v1.json').read_text())
    assert sha(HERE/'PREREG_v1.json')==build['prereg_sha256']
    parent=ROOT/build['records'][0]['path']
    assert sha(parent)==build['source_sha256']
    tree=ast.parse(parent.read_text())
    defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    aa={t.id:n.value for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    edge=ast.literal_eval(aa['_EDGE_LITERAL'])
    context=ast.literal_eval(aa['_MESSAGE_EDGE_LITERAL'])
    knobs=ast.literal_eval(aa['_GN_EDGE'])
    av,bv=(ast.literal_eval(aa[k]) for k in ('_O20_ECUT_RADICAL_A','_O20_ECUT_RADICAL_B'))
    code=compile(ast.Module(body=[defs[n] for n in NAMES],type_ignores=[]),'<actual-edge-output-FD>','exec')
    def environment(output):
        e=copy.deepcopy(edge);e['output']=list(output)
        env=dict(fnp=fnp,_EDGE_PARAMETERS=None,_MESSAGE_EDGE_PARAMETERS=None,_E2_EDGE_STATE={},
            _E2_LAYER={},_RQD_PS32=[None,{}],_RQD_STATS=dict(hits=0),_GN_EDGE=knobs,
            _O20_ECUT_RADICAL_ON=True,_ECUT_STANDIN=True,_O20_ECUT_RADICAL_A=av,
            _O20_ECUT_RADICAL_B=bv,_ECUT_STATS=dict(layers=0,kept=0,full=0),
            _VMFR_CM=8,_O20_EDGE_HIDDEN=8,_EDGE_LITERAL=e,_MESSAGE_EDGE_LITERAL=copy.deepcopy(context))
        exec(code,env);env['_gn_message_edge_source_base']=env['_g5_message_edge_base']
        return env
    rng=np.random.default_rng(20261021)
    cases=[]
    for n,ni,nj,layer in [(8,3,5,10),(16,16,13,14)]:
        inputs=dict(mu=rng.normal(0,.2,n),sigma=rng.uniform(.5,1.5,n),alpha=rng.normal(0,.7,n),
            pdf=rng.uniform(.08,.4,n),gate=rng.uniform(.1,.9,n),pre_k3=rng.normal(0,.02,n),
            pre_k4=rng.normal(0,.01,n),pre_cov=rng.normal(0,.05,(n,n)),
            K21=rng.normal(0,.01,(n,n)),messages=rng.normal(0,1.2,(n,24)))
        records=[dict(name='base',output=edge['output'])]+build['records']
        values={};tariffs={}
        for rec in records:
            env=environment(rec['output'])
            with fs.BudgetContext(flop_budget=2**35,quiet=True) as ctx:
                native={k:fnp.asarray(v.astype('f4')) for k,v in inputs.items()}
                msg=native['messages'];gk=dict(MSG=msg,MSG3=msg.reshape(n,3,8).transpose(1,0,2))
                env['_e2_gnn_consts']=lambda P,width,gk=gk:gk
                env['_e2_messages']=lambda k,W,state:k['MSG3']
                P=dict(msg_w=fnp.zeros((24,64),dtype=fnp.float32))
                aux={k:native[k] for k in ('alpha','pdf','gate','sigma','pre_k3','pre_k4','pre_cov')}
                I,J=fnp.arange(ni,dtype=fnp.int64),fnp.arange(nj,dtype=fnp.int64)
                c=env['_message_edge_parameters']()
                args=(layer,native['mu'],native['K21'],aux,None,None,P,c)
                ys=[];bills=[];calls=[]
                for phase in ['cold','warm']:
                    env['_E2_LAYER']=dict(ecut_rows=I,ecut_cols=J)
                    before,count=ctx.flops_used,len(ctx.op_log)
                    y=env['_message_edge_source'](*args)
                    bills.append(ctx.flops_used-before);calls.append(len(ctx.op_log)-count)
                    ys.append(np.asarray(y).copy())
                np.testing.assert_array_equal(ys[0],ys[1])
                assert np.isfinite(ys[0]).all()
                env['_EDGE_PARAMETERS']=env['_MESSAGE_EDGE_PARAMETERS']=None
                env['_E2_EDGE_STATE'].clear()
                c2=env['_message_edge_parameters']()
                np.testing.assert_array_equal(np.asarray(env['_message_edge_source'](*args[:-1],c2)),ys[0])
                values[rec['name']]=ys[0]
                tariffs[rec['name']]=dict(flops=bills,calls=calls)
        assert all(v==tariffs['base'] for v in tariffs.values())
        y0=values['base'];directions=[]
        for j in range(8):
            ym=values[f'edge8_terminal_fd_c{j}_minus_v1']
            yp=values[f'edge8_terminal_fd_c{j}_plus_v1']
            first=np.linalg.norm(yp-ym)
            second=np.linalg.norm(yp+ym-2*y0)
            assert first>1e-10
            assert second<=1e-8+2e-5*first,(n,j,first,second)
            directions.append(dict(index=j,first_difference_norm=float(first),second_difference_norm=float(second)))
        cases.append(dict(n=n,rows=ni,cols=nj,layer=layer,all16_live=True,tariff=tariffs['base'],
            all16_same_tariff=True,cold_warm_reset_byte_identical=True,directions=directions))
    out=dict(utc=utc(),successful=True,status='OFFICIAL_LITERAL_OUTPUT8_BINDING_TARIFF_PASS',
        official_version=fs.__version__,source_sha256=sha(parent),plan_sha256=sha(HERE/'PLAN_v1.json'),
        prereg_sha256=sha(HERE/'PREREG_v1.json'),checker_sha256=sha(__file__),cases=cases,
        no_accuracy_claim=True,no_truth_read=True)
    with (HERE/'TINY_OFFICIAL_v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(dict(utc=utc(),successful=True,cases=len(cases),receipt_sha256=sha(HERE/'TINY_OFFICIAL_v2.json'))))
if __name__=='__main__':main()

