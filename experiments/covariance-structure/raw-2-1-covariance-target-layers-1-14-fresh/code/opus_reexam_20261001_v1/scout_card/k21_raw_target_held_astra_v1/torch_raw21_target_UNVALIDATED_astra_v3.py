"""HELD private Torch full-target raw-R21 hypothesis. No imports of models.

Configure after the request's original configure_cutsim. Disabled mode delegates
the unchanged cutsim/M installer. Enabled source insertion precedes target users.
"""
import hashlib
import importlib.util
import types
from pathlib import Path


def enable(E):
    path=Path(__file__).with_name('isolated_raw21_astra_v1.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == KERNEL_SHA256
    spec=importlib.util.spec_from_file_location('private_raw21_math',path)
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    original_install=E.M.install
    state={'mode':0,'request_id':None,'events':[],'active':False}

    def configure(request):
        mode=request.get('astra_raw21_target',0)
        assert type(mode)is int and mode in (0,1)
        assert request['mode']=='eval' and not request.get('dump15')
        assert not state['active']
        state.update(mode=mode,request_id=request['id'],events=[])
        if not mode:return
        # This source covers the exact current C law, with RC in the card and
        # KC later in state_out. Other covariance delta families are excluded.
        ncar=request.get('ncar');assert isinstance(ncar,list)
        pbase=[x for x in ncar if x.get('module')=='pbase'];assert len(pbase)==1
        assert pbase[0].get('gsk')==1 and pbase[0].get('gk4')==1
        assert not any(pbase[0].get(k,0)for k in ('gbk','so2m','so2c','so2t','so2f'))
        cut=E._otr_cut_runtime_v1;module=cut['module']
        assert module is not None and cut['spec']['opts'].get('enabled',True)
        def install_cuts(cd,opts=None):
            result=module.install(cd,opts=opts)
            anchor="    C = self.h('cov_out', layer, C, ctx)\n"
            source=cd._cutsim_class_source;assert source.count(anchor)==1
            extra="    target = self._astra_raw21_target(layer, target, host, delta, ctx['C_edge'], C)\n"
            cd._cutsim_class_source=source.replace(anchor,anchor+extra)
            return result
        cut['module']=types.SimpleNamespace(install=install_cuts,install_m=module.install_m)

    def install(cd):
        if not state['mode']:return original_install(cd)
        assert not E.M.CV
        assert all(not bool(E.M.PL.EPS[f][1:15].abs().sum()>0)for f in E.M.CV_FAMILIES)
        t=cd.t;context={};events=state['events']
        cd._astra_raw21_context=context;cd.astra_raw21_events=events
        def target(layer,K,h,d,C0,C):
            if not 1<=int(layer)<=14:return K
            assert not context and cd.knobs['recenter_part']=='all'
            kc=E.M.PL.EPS['kc'][layer].to(C.dtype)
            # Exact original M.state_out expression, evaluated privately here
            # solely for the new target law; the actual carried covariance is
            # still produced by the unchanged original state_out hook.
            C1=C+kc*(t.outer(h,d)+t.outer(d,h)+t.outer(d,d))
            answer=math.raw21_actual_covariance(t,K,h,d,C0,C1)
            context.update(layer=int(layer),expected_covariance=C1)
            state['active']=True
            events.append({'layer':int(layer),'shape':list(K.shape),'actual_covariance_law':True})
            return answer
        cd._astra_raw21_target=target
        result=original_install(cd)
        original_out=cd.hooks['state_out']
        def state_out(layer,out,ctx):
            result=original_out(layer,out,ctx)
            if 1<=int(layer)<=14:
                assert context.get('layer')==int(layer)
                assert t.equal(result[1],context['expected_covariance']), 'Uncovered covariance hook: stop'
                context.clear();state['active']=False
            return result
        cd.hooks['state_out']=state_out
        step0=cd.step
        def step(layer,value):
            assert not context
            try:
                answer=step0(layer,value)
                assert not context
                return answer
            finally:context.clear();state['active']=False
        cd.step=step
        return result

    configure.route_report=lambda:dict(mode=state['mode'],events=[dict(x)for x in state['events']],retired=not state['active'])
    E.M.install=install
    return configure


# Replaced only by the source-only composer before any request can use this file.
KERNEL_SHA256='d4dad9c421bc11a808641147f5b3ad1eea560e990cb84df269d357e81b669fdb'
