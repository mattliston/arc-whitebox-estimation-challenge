import ast,json,math,resource
from common import *
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from capture import observed
    assert all(x['num_threads']==1 for x in threadpool_info());verify()
    funcs,proof=observed();tree=ast.parse(SOURCE.read_text())
    lit=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_EDGE_LITERAL' for t in n.targets))
    rng=np.random.default_rng(26092070);records=[]
    for n in (5,13):
        a=rng.normal(size=n);s=np.exp(.2*rng.normal(size=n));p=np.exp(-.5*a*a)/math.sqrt(2*math.pi);g=np.array([.5*(1+math.erf(x/math.sqrt(2))) for x in a])
        C=rng.normal(size=(n,n))*.01;np.fill_diagonal(C,s*s)
        k3=rng.normal(size=n)*s**3*.01;k4=rng.normal(size=n)*s**4*.01
        S=rng.normal(size=(n,n))*.001;np.fill_diagonal(S,k3)
        m=s*(a*g+p)+.001*rng.normal(size=n)
        aux=dict(alpha=a,pdf=p,gate=g,sigma=s,pre_cov=C,pre_k3=k3,pre_k4=k4)
        env=dict(fnp=fnp,_EDGE_LITERAL=lit,_EDGE_PARAMETERS=None,_CIS_EDGE={})
        exec(funcs['_edge_source'],env)
        with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
            am={k:fnp.asarray(v) for k,v in aux.items()};result=env['_edge_source'](8,fnp.asarray(m),fnp.asarray(S),am)
        d={k:np.asarray(v) for k,v in env['_CIS_EDGE'][8].items()};t={k:np.asarray(v) for k,v in env['_EDGE_PARAMETERS'].items()}
        B=basis(d,t);actual=np.asarray(result);ref=B@t['output'].astype(np.float64)
        relative=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(actual),1e-30));assert relative<=3e-6,relative
        assert np.array_equal(B,B.transpose(1,0,2))
        perm=rng.permutation(n);dp={k:(v[np.ix_(perm,perm)] if k in ('R','S','correction') else v[perm]) for k,v in d.items()}
        bp=basis(dp,t);perr=float(np.max(np.abs(bp-B[perm][:,perm])));assert perr<=2e-10*max(1.,float(np.max(np.abs(B))))
        q=rng.normal(size=16);U=rng.normal(size=(n,n))
        X=B.reshape(-1,16);G=X.T@X/(n*n);b=X.T@U.ravel()/(n*n);e=float(np.mean(U*U))
        explicit=float(np.mean((U-B@q)**2));quad=float(e-2*q@b+q@G@q)
        assert abs(explicit-quad)<=2e-10*max(explicit,1e-30)
        # Independent scalar entry loop for the real-linear edge basis.
        h=(d['F']@t['left'])[:,None,:]+(d['F']@t['right'])[None,:,:]+t['bias']
        h=h+d['R'][:,:,None]*t['edge'][0]+d['S'][:,:,None]*t['edge'][1]+d['S'].T[:,:,None]*t['edge'][2]
        H=np.tanh(h)
        loop=np.empty_like(B)
        for i in range(n):
            for j in range(n):
                for k in range(16):
                    loop[i,j,k]=.5*(float(H[i,j,k])*float(d['pf'][i])*float(d['gf'][j])+float(H[j,i,k])*float(d['pf'][j])*float(d['gf'][i]))*float(d['sf'][i])*float(d['sf'][j])*float(t['cov_factor'])
        assert np.max(np.abs(loop-B))<=2e-10*max(1.,float(np.max(np.abs(B))))
        # Literal Gaussian-gain covariance derivative; asymmetric C allowed.
        W=rng.normal(size=(n,n))/math.sqrt(n);ein=rng.normal(size=n)*.01;E=rng.normal(size=(n,n))*.01;E=(E+E.T)*.5
        mu=a*s;mg=s*(a*g+p);pre=dict(sigma=s,alpha=a,pdf=p,gate=g,mg=mg,C_pre=C)
        V=W.T@E@W;u=W.T@ein
        def gaussian_cov(muc,Cc):
            v=np.diag(Cc);ss=np.sqrt(v);aa=muc/ss;pp=np.exp(-aa*aa/2)/math.sqrt(2*math.pi);gg=np.array([.5*(1+math.erf(x/math.sqrt(2))) for x in aa]);mm=muc*gg+ss*pp
            out=Cc*(gg[:,None]*gg[None,:]);np.fill_diagonal(out,(muc*muc+v)*gg+muc*ss*pp-mm*mm)
            return (out+out.T)*.5
        step=1e-5;fd=(gaussian_cov(mu+step*u,C+step*V)-gaussian_cov(mu-step*u,C-step*V))/(2*step)
        tan=tangent_cov(pre,ein,E,W);trel=float(np.linalg.norm(fd-tan)/np.linalg.norm(tan));assert trel<=2e-7,trel
        records.append(dict(n=n,native_reconstruction_relative=relative,permutation_maxabs=perr,tangent_fd_relative=trel,flops=ctx.summary_dict()['flops_used']))
    write(HERE/'unit.json',dict(passed=True,records=records,source_proof=proof,targets_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(passed=True,records=records)),flush=True)
if __name__=='__main__':main()
