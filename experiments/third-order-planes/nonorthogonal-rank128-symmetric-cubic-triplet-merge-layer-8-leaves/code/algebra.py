"""Factored ALS and explicit symmetric triplet contractions, FP64 research."""
import itertools,math
import numpy as np

class Products:
    def __init__(self):self.records=[]
    def mm(self,a,b,tag):
        assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[0]
        self.records.append(dict(tag=tag,shapes=[list(a.shape),list(b.shape)],fp64_leading_billed_flops=4*int(a.shape[0])*int(a.shape[1])*int(b.shape[1])))
        return a@b

def balance(factors):
    norms=[np.linalg.norm(a,axis=0) for a in factors]
    assert all(np.isfinite(n).all() and (n>0).all() for n in norms)
    target=np.exp(sum(np.log(n) for n in norms)/3)
    scales=[target/n for n in norms]
    return [a*s[None,:] for a,s in zip(factors,scales)],scales

def fit(f,lam,rank,products,seed=2026092005):
    d,m=f.shape;assert rank<=d
    rng=np.random.Generator(np.random.PCG64DXSM(seed))
    factors=[np.zeros((d,rank)),np.linalg.qr(rng.standard_normal((d,rank)),mode='reduced')[0],np.linalg.qr(rng.standard_normal((d,rank)),mode='reduced')[0]]
    cross=[None]+[products.mm(f.T,x,'construction_initial_cross') for x in factors[1:]]
    reports=[]
    for sweep in range(2):
        for owner in range(3):
            others=[i for i in range(3) if i!=owner];i,j=others
            mtt=products.mm(f,lam[:,None]*cross[i]*cross[j],'construction_mttkrp')
            gi=products.mm(factors[i].T,factors[i],'construction_factor_gram')
            gj=products.mm(factors[j].T,factors[j],'construction_factor_gram')
            g=gi*gj;trace=float(np.trace(g));rho=1e-8*trace/rank
            assert np.isfinite(trace) and trace>0 and rho>0
            regular=g+rho*np.eye(rank)
            factors[owner]=np.linalg.solve(regular,mtt.T).T
            assert np.isfinite(factors[owner]).all()
            residual=float(np.linalg.norm(products.mm(factors[owner],regular,'qualification_normal_equation')-mtt)/max(np.linalg.norm(mtt),1e-300))
            assert residual<=1e-9
            cross[owner]=products.mm(f.T,factors[owner],'construction_updated_cross')
            reports.append(dict(sweep=sweep+1,mode=owner,regularization=rho,normal_equation_relative_residual=residual))
        factors,scales=balance(factors)
        cross=[x*s[None,:] for x,s in zip(cross,scales)]
    return factors,reports

def cp_query(p,lam,a,b,products,tag):
    aa=products.mm(p.T,a,tag+'_a');bb=products.mm(p.T,b,tag+'_b')
    return products.mm(p,lam[:,None]*aa*bb,tag+'_read')

def triplet_query(factors,a,b,products,tag):
    aa=[products.mm(x.T,a,tag+'_a') for x in factors]
    bb=[products.mm(x.T,b,tag+'_b') for x in factors]
    answer=np.zeros((factors[0].shape[0],a.shape[1]))
    for i in range(3):
        j,k=[x for x in range(3) if x!=i]
        answer+=products.mm(factors[i],(aa[j]*bb[k]+aa[k]*bb[j])/6,tag+'_read')
    return answer

def triplet_plane(factors,products,tag):
    a,b,c=factors
    return (products.mm(a*b,c.T,tag)+products.mm(a*c,b.T,tag)+products.mm(b*c,a.T,tag))/3

def norms(f,lam,factors,products,block=256):
    vals=[];absolute=[]
    for start in range(0,f.shape[1],block):
        g=products.mm(f[:,start:start+block].T,f,'diagnostic_target_norm')
        terms=(g*g*g)*lam[start:start+block,None]*lam[None,:]
        vals.append(float(terms.sum()));absolute.append(float(np.abs(terms).sum()))
    norm=math.fsum(vals)
    cross=[products.mm(f.T,x,'diagnostic_cross_inner') for x in factors]
    inner=float(np.sum(lam[:,None]*cross[0]*cross[1]*cross[2]))
    grams={(i,j):products.mm(factors[i].T,factors[j],'diagnostic_factor_gram') for i in range(3) for j in range(3)}
    unsym=float(np.sum(grams[0,0]*grams[1,1]*grams[2,2]))
    sym=math.fsum(float(np.sum(grams[0,p[0]]*grams[1,p[1]]*grams[2,p[2]])) for p in itertools.permutations(range(3)))/6
    return dict(target_norm_squared=norm,target_gram_absolute_sum=math.fsum(absolute),target_merged_inner=inner,unsymmetrized_norm_squared=unsym,symmetrized_norm_squared=sym,unsymmetrized_error_squared=norm+unsym-2*inner,symmetrized_error_squared=norm+sym-2*inner)
