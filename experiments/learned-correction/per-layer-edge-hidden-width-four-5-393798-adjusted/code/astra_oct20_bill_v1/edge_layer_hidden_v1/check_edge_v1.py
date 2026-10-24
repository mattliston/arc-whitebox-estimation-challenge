"""Official per-layer folded-edge oracle, independent caches, tariff, and input ownership."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
import argparse, ast, copy, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp
assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
COMMON=('_E2Hold','_g5_sub','_g5_chain','_e2_ms','_ps_pre32','_ps_k2132','_ps32','_ecut_raw')
ap=argparse.ArgumentParser()
for key in ('fit','build','out'):ap.add_argument('--'+key,type=Path,required=True)
a=ap.parse_args();fit=read(a.fit);b=read(a.build);width=b['width']
sp=Path(b['outputs']['science']['path']);parent=Path(b['outputs']['science']['parent_path'])
assert sha(sp)==b['outputs']['science']['sha256'] and sha(parent)==fit['teacher_source_sha256']
assert sp.read_text().startswith(parent.read_text()) and sha(a.fit)==b['fit_sha256']
old={n.name:n for n in ast.parse(parent.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new={n.name:n for n in ast.parse(sp.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for key in COMMON:assert ast.dump(old[key])==ast.dump(new[key]),key
for key in old:
    if key not in ('_g5_message_edge_base','_message_edge_parameters'):
        assert ast.dump(old[key])==ast.dump(new[key]),key
rows={r['layer']:r for r in fit['results'] if r['width']==width}
assert set(rows)==set(range(8,15))
mp=Path(fit['captures'][0]['path']).parent/'METADATA_v1.json'
assert sha(mp)==fit['captures'][0]['metadata_sha256'];meta=read(mp)
aa,bb=(float(np.float32(fit['activation'][k])) for k in ('a','b'))
def environment(student):
    env=dict(fnp=fnp,_EDGE_PARAMETERS=None,_MESSAGE_EDGE_PARAMETERS=None,_E2_EDGE_STATE={},_E2_LAYER={},
        _RQD_PS32=[None,{}],_RQD_STATS=dict(hits=0),_O20_ECUT_RADICAL_ON=True,_ECUT_STANDIN=True,
        _O20_ECUT_RADICAL_A=aa,_O20_ECUT_RADICAL_B=bb,_ECUT_STATS=dict(layers=0,kept=0,full=0),
        _VMFR_CM=8,_O20_EDGE_HIDDEN=width if student else 14,
        _EDGE_LITERAL=copy.deepcopy(rows[8]['edge_literal'] if student else meta['edge_literal']),
        _MESSAGE_EDGE_LITERAL=copy.deepcopy(rows[8]['message_edge_literal'] if student else meta['message_edge_literal']))
    defs=new if student else old
    names=COMMON+(('_pl_edge_parameters','_pl_edge_consts') if student else ('_e2_edge_consts',))+('_message_edge_parameters','_g5_message_edge_base')
    if student:env.update(_PL_EDGE_LITERALS={k:copy.deepcopy(r['edge_literal']) for k,r in rows.items()},
        _PL_MESSAGE_LITERALS={k:copy.deepcopy(r['message_edge_literal']) for k,r in rows.items()},
        _PL_EDGE_PARAMETERS={},_PL_EDGE_CONSTS={})
    exec(compile(ast.Module(body=[defs[k] for k in names],type_ignores=[]),'<live-edge-functions>','exec'),env)
    env['_gn_message_edge_source_base']=env['_g5_message_edge_base']
    return env
rng=np.random.default_rng(20261021);cases=[]
for n,ni,nj,mult,layers in [(8,3,5,1.,list(range(8,15))),(16,16,16,7.,[8,11,14]),(1024,31,47,3.,[8,14,8])]:
    inputs=dict(mu=rng.normal(0,.2,n),sigma=rng.uniform(.5,1.5,n),alpha=rng.normal(0,.7,n),
        pdf=rng.uniform(.08,.4,n),gate=rng.uniform(.1,.9,n),pre_k3=rng.normal(0,.02,n),
        pre_k4=rng.normal(0,.01,n),pre_cov=rng.normal(0,.05,(n,n)),K21=rng.normal(0,.01,(n,n)),
        messages=rng.normal(0,.8*mult,(n,24)))
    inputs={k:v.astype('f4') for k,v in inputs.items()}
    ii=np.arange(ni);jj=np.arange(nj);teacher_pre=[];baseline=None
    with fs.BudgetContext(flop_budget=2**35,quiet=True) as ctx:
        vals={k:fnp.asarray(v) for k,v in inputs.items()}
        P=dict(msg_w=fnp.zeros((24,64),dtype=fnp.float32))
        aux={k:vals[k] for k in ('alpha','pdf','gate','sigma','pre_k3','pre_k4','pre_cov')}
        I,J=fnp.arange(ni,dtype=fnp.int64),fnp.arange(nj,dtype=fnp.int64)
        def configure(env):
            msg=fnp.asarray(inputs['messages'].copy())
            gk=dict(MSG=msg,MSG3=msg.reshape(n,3,8).transpose(1,0,2))
            env['_e2_gnn_consts']=lambda P,width:gk
            env['_e2_messages']=lambda k,W,state:k['MSG3']
            env['_E2_LAYER']=dict(ecut_rows=I,ecut_cols=J)
            return env
        teacher=configure(environment(False));student=configure(environment(True))
        original_raw=teacher['_ecut_raw']
        def observe(left,right,t,k,pf,gf,hold):
            l,r,bias=(np.asarray(v).copy() for v in (left,right,t['bias']))
            R,S=(np.asarray(k[v]).copy() for v in ('R','S'));edge=np.asarray(t['edge']).copy()
            z=(l[ii]+bias)[:,None,:]+r[jj]
            z+=R[np.ix_(ii,jj)][:,:,None]*edge[0]
            z+=S[np.ix_(ii,jj)][:,:,None]*edge[1]
            z+=S[np.ix_(jj,ii)].T[:,:,None]*edge[2]
            teacher_pre.append(z.copy())
            return original_raw(left,right,t,k,pf,gf,hold)
        def run(env,layer):
            context=env['_message_edge_parameters']()
            args=(layer,vals['mu'],vals['K21'],aux,None,None,P,context)
            before,count=ctx.flops_used,len(ctx.op_log)
            y=env['_gn_message_edge_source_base'](*args)
            return np.asarray(y).copy(),ctx.flops_used-before,len(ctx.op_log)-count
        run(teacher,8)
        _,teacher_bill,teacher_calls=run(teacher,8)
        teacher['_ecut_raw']=observe
        run(teacher,8)
        teacher_z=teacher_pre[-1];seen={};subcases=[]
        for layer in layers:
            cold=run(student,layer);warm=run(student,layer)
            np.testing.assert_array_equal(cold[0],warm[0]);assert np.isfinite(warm[0]).all()
            if layer in seen:np.testing.assert_array_equal(seen[layer],warm[0])
            else:seen[layer]=warm[0].copy()
            r=rows[layer]
            z=teacher_z@np.asarray(r['Q'],dtype='f4')+np.asarray(r['offset'],dtype='f4')
            h=np.clip(z/(np.float32(aa)+np.sqrt(np.float32(bb)+z*z)),-1,1)
            block=(h@np.asarray(r['output'],dtype='f4'))*inputs['pdf'][ii,None]*inputs['gate'][None,jj]
            raw=np.zeros((n,n),dtype='f4');raw[np.ix_(ii,jj)]=block
            oracle=(.5*(raw+raw.T))*np.outer(inputs['sigma'],inputs['sigma'])*np.float32(r['edge_literal']['cov_factor'])
            error=float(np.max(np.abs(warm[0]-oracle)));scale=max(float(np.max(np.abs(oracle))),1e-10)
            assert error<2e-9+5e-5*scale,(n,layer,error,scale)
            saving=teacher_bill-warm[1];expected=(14-width)*(150*n+5*ni+4*nj+16*ni*nj)
            assert saving==expected,(n,layer,saving,expected)
            assert teacher_calls==warm[2],(teacher_calls,warm[2])
            subcases.append(dict(layer=layer,oracle_abs_error=error,oracle_scale=scale,cold_bill=cold[1],
                warm_bill=warm[1],warm_saving=saving,expected_saving=expected,cold_calls=cold[2],warm_calls=warm[2]))
        assert len({id(v['R']) for v in student['_PL_EDGE_CONSTS'].values()})==len(student['_PL_EDGE_CONSTS'])
        assert len({id(v['S']) for v in student['_PL_EDGE_CONSTS'].values()})==len(student['_PL_EDGE_CONSTS'])
        scratch_bytes=sum(int(v['R'].size+v['S'].size)*4 for v in student['_PL_EDGE_CONSTS'].values())
        for key in inputs:
            if key!='messages':np.testing.assert_array_equal(np.asarray(vals[key]),inputs[key])
        student['_PL_EDGE_PARAMETERS'].clear();student['_PL_EDGE_CONSTS'].clear()
        rebuilt=run(student,layers[0]);np.testing.assert_array_equal(rebuilt[0],seen[layers[0]])
        t,context=student['_pl_edge_parameters'](8);good=context['left']
        context['left']=fnp.zeros((24,14),dtype=fnp.float32)
        try:run(student,8)
        except ValueError as err:assert str(err)=='Message covariance draft shape mismatch'
        else:raise AssertionError('shape guard absent')
        context['left']=good
        for badlayer in (7,15):
            try:run(student,badlayer)
            except ValueError as err:assert str(err)=='Frozen covariance edge layer'
            else:raise AssertionError('layer guard absent')
        cases.append(dict(n=n,rows=ni,cols=nj,teacher_warm_bill=teacher_bill,teacher_warm_calls=teacher_calls,
            layers=subcases,cache_rebuild_byte_exact=True,layer_permutation_byte_exact=True,
            private_scratch_bytes=scratch_bytes,inputs_unchanged=True,live_shape_and_layer_guards=True))
receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
    successful=True,script_sha256=sha(__file__),source_sha256=sha(sp),fit_sha256=sha(a.fit),
    official_version=fs.__version__,width=width,cases=cases,no_native_accuracy_claim=True)
with a.out.open('x') as out:json.dump(receipt,out,indent=2);out.write('\n')
print(json.dumps(dict(successful=True,width=width,cases=len(cases),source_sha256=sha(sp))))

