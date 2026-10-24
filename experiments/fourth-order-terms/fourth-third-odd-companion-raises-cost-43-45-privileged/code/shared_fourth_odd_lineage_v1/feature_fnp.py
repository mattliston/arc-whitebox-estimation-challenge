"""Fixed empirical odd companion, with paid parent-kernel transport."""
import ast
import hashlib
import flopscope.numpy as fnp


def epsilon_from_aux(aux):
    return -aux['alpha']*aux['pdf']/(aux['sigma']*aux['sigma'])


def pair_gamma(epsilon, selected, t, s):
    gap = t.astype(fnp.float64)-s.astype(fnp.float64)
    nonzero = gap != 0
    return fnp.where(nonzero, epsilon[selected]/fnp.where(nonzero,gap,1.),0.)


def advance_u(mm, layer, w32, previous, u, pp, old_columns, basis, pre_basis):
    inherited = int(u.shape[1])
    rows = int(pp.shape[0])
    if inherited:
        old_active = u[previous,:]
        if 5 <= layer < 15:
            assert basis is not None and pre_basis is not None
            factors = mm(basis.T, old_active[:,:old_columns])
            old = mm(pre_basis, factors)
            young = mm(w32[previous,:].T, old_active[:,old_columns:])
            pushed = fnp.concatenate([old,young],axis=1)
        else:
            pushed = mm(w32[previous,:].T,old_active)
    else:
        pushed = fnp.zeros((rows,0),dtype=fnp.float32)
    newborn = int(pp.shape[1])-inherited
    assert newborn >= 0
    return fnp.concatenate([pushed,fnp.zeros((rows,newborn),dtype=fnp.float32)],axis=1)


def gate_u(upre, pp, gate, derivative, alive):
    g = gate.astype(fnp.float32)
    d = derivative.astype(fnp.float32)
    mask = alive.astype(fnp.float32)
    fnp.multiply(upre,g[:,None],out=upre)
    emission = pp*pp
    fnp.multiply(emission,d[:,None],out=emission)
    fnp.add(upre,emission,out=upre)
    fnp.multiply(upre,mask[:,None],out=upre)
    return upre


def odd_diagonal(upre, pp, cohorts):
    total = fnp.zeros(int(pp.shape[0]),dtype=fnp.float64)
    stop = 0
    for start,gamma in cohorts:
        count = int(gamma.shape[0])
        assert start == stop
        stop = start+2*count
        x = pp[:,start:start+count].astype(fnp.float64)
        y = pp[:,start+count:stop].astype(fnp.float64)
        a = upre[:,start:start+count].astype(fnp.float64)
        b = upre[:,start+count:stop].astype(fnp.float64)
        difference = (a-b)*(x*x)+b*(x-y)*(x+y)
        total = total+fnp.sum(difference*gamma[None,:],axis=1)
    assert stop == int(pp.shape[1])
    return 1.5*total


def mean_feature(aux, delta_k3):
    return -aux['alpha']*aux['pdf']*delta_k3/(6*aux['sigma']*aux['sigma'])


def observed_push(source):
    tree = ast.parse(source)
    node = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'][-1]
    original = ast.get_source_segment(source,node)
    before = "    state['pre_basis'] = U\n"
    after = before+"    state['_odd4_input_basis'] = active_basis\n"
    assert original.count(before)==1
    changed = original.replace(before,after)
    assert changed.replace(after,before)==original
    return changed,dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                        observed_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                        inverse_byte_exact=True,extra_parent_numeric_operations=0)


class OddLineageTap:
    NAMES = ('_nonlinear_first_gaussian','nonlinear','_shared_old384_push',
             '_shared_old384_commit','nonlinear_scalar','gnn_step')

    def __init__(self,mod,ctx,observed):
        self.mod,self.ctx = mod,ctx
        self.original = {n:getattr(mod,n) for n in self.NAMES}
        exec(compile(observed,'observed_odd_push.py','exec'),mod.__dict__)
        self.basis_push = mod._shared_old384_push
        mod._shared_old384_push = self.original['_shared_old384_push']
        self.extra_flops = 0
        self.cohorts = []
        self.pending = None
        self.layer = -1
        self.saved = None
        self.aux = None
        self.held = {}
        self.u = self.measure(fnp.zeros,(1024,0),dtype=fnp.float32)

    def measure(self,fun,*args,**kwargs):
        before = self.ctx.summary_dict()['flops_used']
        answer = fun(*args,**kwargs)
        self.extra_flops += self.ctx.summary_dict()['flops_used']-before
        return answer

    def remember(self,result):
        assert self.pending is None and 0 <= self.layer <= 14
        aux = result[-1]
        self.pending = (self.layer,self.measure(epsilon_from_aux,aux))
        self.aux = {key:aux[key] for key in ('pdf','sigma','alpha')}
        return result

    def first(self,*args,**kwargs):
        return self.remember(self.original['_nonlinear_first_gaussian'](*args,**kwargs))

    def pair(self,*args,**kwargs):
        return self.remember(self.original['nonlinear'](*args,**kwargs))

    def push(self,layer,w32,p,previous,newborn,lam,state):
        assert self.saved is None
        self.layer = layer
        answer = self.basis_push(layer,w32,p,previous,newborn,lam,state)
        if layer:
            assert self.pending is not None and self.pending[0]==layer-1
            start,selected,unused_base,t,s = newborn
            assert start==sum(2*int(g.shape[0]) for _,g in self.cohorts)
            gamma = self.measure(pair_gamma,self.pending[1],selected,t,s)
            assert start+2*int(gamma.shape[0])==int(lam.shape[0])
            self.cohorts.append((start,gamma))
            self.pending = None
        self.saved = (w32,previous,answer,state)
        return answer

    def advance(self):
        w32,previous,pp,state = self.saved
        basis = state.pop('_odd4_input_basis',None)
        return self.measure(advance_u,self.mod._shared_old384_mm,self.layer,w32,previous,
                            self.u,pp,int(state['columns']),basis,state['pre_basis'])

    def commit(self,layer,gate,alive,selected,state):
        assert layer==self.layer and self.saved[3] is state
        pp = self.saved[2]
        upre = self.advance()
        derivative = self.measure(fnp.divide,self.aux['pdf'],self.aux['sigma'])
        self.u = self.measure(gate_u,upre,pp,gate,derivative,alive)
        self.saved = None
        return self.original['_shared_old384_commit'](layer,gate,alive,selected,state)

    def scalar(self,*args,**kwargs):
        answer = self.original['nonlinear_scalar'](*args,**kwargs)
        assert self.layer==15
        self.aux = {key:answer[1][key] for key in ('pdf','sigma','alpha')}
        return answer

    def gnn(self,l,x,sigma,w,state,params):
        result = self.original['gnn_step'](l,x,sigma,w,state,params)
        if l==15:
            upre = self.advance()
            delta = self.measure(odd_diagonal,upre,self.saved[2],self.cohorts)
            self.held['delta_k3'] = delta
            self.held['feature'] = self.measure(mean_feature,self.aux,delta)
            self.held['sigma'] = sigma
            self.saved = None
            self.u = None
        return result

    def attach(self):
        wrappers = (self.first,self.pair,self.push,self.commit,self.scalar,self.gnn)
        for name,value in zip(self.NAMES,wrappers):setattr(self.mod,name,value)

    def detach(self):
        for name,value in self.original.items():setattr(self.mod,name,value)
        assert all(getattr(self.mod,name) is value for name,value in self.original.items())
