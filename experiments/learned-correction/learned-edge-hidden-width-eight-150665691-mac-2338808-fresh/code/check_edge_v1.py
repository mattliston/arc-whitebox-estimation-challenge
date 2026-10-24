"""Tiny official live edge functions: folded width, caches, tariff and numerical oracle."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp
assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES=('_E2Hold','_g5_sub','_g5_chain','_e2_ms','_ps_pre32','_ps_k2132','_ps32',
       '_e2_edge_consts','_message_edge_parameters','_g5_message_edge_base','_ecut_raw')

def main():
    ap=argparse.ArgumentParser()
    for k in ['fit','build','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();fit=json.loads(a.fit.read_text());build=json.loads(a.build.read_text())
    source=Path(build['outputs']['science']['path']);parent=Path(build['outputs']['science']['parent_path'])
    assert sha(source)==build['outputs']['science']['sha256'] and sha(parent)==fit['teacher_source_sha256']
    assert source.read_text().startswith(parent.read_text())
    olddefs={n.name:n for n in ast.parse(parent.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newdefs={n.name:n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert all(ast.dump(olddefs[k])==ast.dump(newdefs[k]) for k in NAMES)
    code=compile(ast.Module(body=[olddefs[k] for k in NAMES],type_ignores=[]),'<actual-edge-functions>','exec')
    mp=Path(fit['captures'][0]['path']).parent/'METADATA_v1.json';assert sha(mp)==fit['captures'][0]['metadata_sha256']
    meta=json.loads(mp.read_text());Q=np.asarray(fit['Q'],dtype='f4');offset=np.asarray(fit['offset'],dtype='f4')
    aa,bb=(float(np.float32(fit['activation'][k])) for k in ['a','b'])
    def environment(hidden):
        env=dict(fnp=fnp,_EDGE_PARAMETERS=None,_MESSAGE_EDGE_PARAMETERS=None,_E2_EDGE_STATE={},_E2_LAYER={},
            _RQD_PS32=[None,{}],_RQD_STATS=dict(hits=0),_O20_ECUT_RADICAL_ON=True,_ECUT_STANDIN=True,
            _O20_ECUT_RADICAL_A=aa,_O20_ECUT_RADICAL_B=bb,_ECUT_STATS=dict(layers=0,kept=0,full=0),
            _VMFR_CM=8,_O20_EDGE_HIDDEN=hidden,
            _EDGE_LITERAL=copy.deepcopy(meta['edge_literal'] if hidden==14 else fit['edge_literal']),
            _MESSAGE_EDGE_LITERAL=copy.deepcopy(meta['message_edge_literal'] if hidden==14 else fit['message_edge_literal']))
        exec(code,env);env['_gn_message_edge_source_base']=env['_g5_message_edge_base']
        return env
    rng=np.random.default_rng(20261020);cases=[]
    for n,ni,nj,mult in [(8,3,5,1.),(12,7,9,3.),(16,16,16,7.)]:
        inputs=dict(mu=rng.normal(0,.2,n),sigma=rng.uniform(.5,1.5,n),alpha=rng.normal(0,.7,n),
            pdf=rng.uniform(.08,.4,n),gate=rng.uniform(.1,.9,n),pre_k3=rng.normal(0,.02,n),
            pre_k4=rng.normal(0,.01,n),pre_cov=rng.normal(0,.05,(n,n)),
            K21=rng.normal(0,.01,(n,n)),messages=rng.normal(0,.8*mult,(n,24)))
        ii=np.arange(ni);jj=np.arange(nj);result={};teacher_pre=None
        for hidden in [14,8]:
            env=environment(hidden)
            with fs.BudgetContext(flop_budget=2**32,quiet=True) as ctx:
                vals={k:fnp.asarray(v.astype('f4')) for k,v in inputs.items()};msg=vals['messages']
                gk=dict(MSG=msg,MSG3=msg.reshape(n,3,8).transpose(1,0,2))
                env['_e2_gnn_consts']=lambda P,width,gk=gk:gk
                env['_e2_messages']=lambda k,W,state:k['MSG3']
                P=dict(msg_w=fnp.zeros((24,64),dtype=fnp.float32))
                aux={k:vals[k] for k in ('alpha','pdf','gate','sigma','pre_k3','pre_k4','pre_cov')}
                I,J=fnp.arange(ni,dtype=fnp.int64),fnp.arange(nj,dtype=fnp.int64)
                context=env['_message_edge_parameters']();assert context['left'].shape==(24,hidden)
                args=(10,vals['mu'],vals['K21'],aux,None,None,P,context)
                bills,calls,ys=[],[],[]
                for phase in ['cold','warm']:
                    env['_E2_LAYER']=dict(ecut_rows=I,ecut_cols=J)
                    before,count=ctx.flops_used,len(ctx.op_log)
                    y=env['_gn_message_edge_source_base'](*args)
                    bills.append(ctx.flops_used-before);calls.append(len(ctx.op_log)-count)
                    ys.append(np.asarray(y).copy())
                np.testing.assert_array_equal(ys[0],ys[1]);assert np.isfinite(ys[-1]).all()
                result[hidden]=dict(bills=bills,calls=calls,output=ys[-1])
                if hidden==14:
                    original_raw=env['_ecut_raw'];saved=[]
                    def observe(left,right,t,k,pf,gf,hold):
                        l,r,b=(np.asarray(v).copy() for v in [left,right,t['bias']])
                        R,S=(np.asarray(k[v]).copy() for v in ['R','S'])
                        e=np.asarray(t['edge']).copy()
                        z=(l[ii]+b)[:,None,:]+r[jj]
                        z+=R[np.ix_(ii,jj)][:,:,None]*e[0]
                        z+=S[np.ix_(ii,jj)][:,:,None]*e[1]
                        z+=S[np.ix_(jj,ii)].T[:,:,None]*e[2]
                        saved.append(z.copy())
                        return original_raw(left,right,t,k,pf,gf,hold)
                    env['_ecut_raw']=observe
                    env['_gn_message_edge_source_base'](*args)
                    teacher_pre=saved[0]
                else:
                    bad=dict(context,left=fnp.zeros((24,14),dtype=fnp.float32))
                    try:env['_gn_message_edge_source_base'](*args[:-1],bad)
                    except ValueError as error:assert str(error)=='Message covariance draft shape mismatch'
                    else:raise AssertionError('live shape guard did not run')
                    env['_EDGE_PARAMETERS']=env['_MESSAGE_EDGE_PARAMETERS']=None;env['_E2_EDGE_STATE'].clear()
                    rebuilt=env['_message_edge_parameters']()
                    yy=env['_gn_message_edge_source_base'](*args[:-1],rebuilt)
                    np.testing.assert_array_equal(np.asarray(yy),ys[-1])
        z=teacher_pre@Q+offset
        h=np.clip(z/(np.float32(aa)+np.sqrt(np.float32(bb)+z*z)),-1,1)
        block=(h@np.asarray(fit['output'],dtype='f4'))*inputs['pdf'].astype('f4')[ii,None]*inputs['gate'].astype('f4')[None,jj]
        raw=np.zeros((n,n),dtype='f4');raw[np.ix_(ii,jj)]=block
        sigma=inputs['sigma'].astype('f4')
        oracle=(.5*(raw+raw.T))*np.outer(sigma,sigma)*np.float32(fit['edge_literal']['cov_factor'])
        diff=float(np.max(np.abs(result[8]['output']-oracle)))
        scale=max(float(np.max(np.abs(oracle))),1e-10)
        assert diff < 2e-9+5e-5*scale,(n,diff,scale)
        saving=result[14]['bills'][1]-result[8]['bills'][1]
        expected=6*(150*n+5*ni+4*nj+16*ni*nj)
        assert saving==expected,(saving,expected)
        assert result[14]['calls']==result[8]['calls']
        cases.append(dict(n=n,rows=ni,cols=nj,teacher_bills=result[14]['bills'],student_bills=result[8]['bills'],
            teacher_calls=result[14]['calls'],student_calls=result[8]['calls'],warm_saving=saving,
            tariff_formula_exact=True,fold_oracle_abs_error=diff,fold_oracle_scale=scale,
            same_cache_rebuild_bytes=True,live_shape_guard=True,reference_uses_teacher_preactivation=True))
    receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
        successful=True,status='TINY_OFFICIAL_FOLDED_EDGE_PASS',script_sha256=sha(__file__),source_sha256=sha(source),
        fit_sha256=sha(a.fit),official_version=fs.__version__,cases=cases,
        note='Approximate hidden8 distillation; tiny folded arithmetic, tariff and binding checks are not native accuracy or runtime evidence.')
    with a.out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
