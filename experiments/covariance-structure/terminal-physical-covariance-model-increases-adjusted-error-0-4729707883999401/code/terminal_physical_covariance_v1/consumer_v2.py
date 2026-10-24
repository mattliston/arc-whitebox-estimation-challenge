"""Post14 physical covariance proposal consumer; native fidelity is separately gated.

Frozen prefix, zero-diagonal D: CP births, lambda, m14, H14, b14 and radial
remain fixed. C14 and its derived star14 change. The current 48/32 star
projection is recomputed, including QR/power/SVD. Native baselines anchor
star rounding differences. Variance uses the actual native FNP forward and
a declared algebraic straight-through VJP, not a rounded-kernel derivative.
"""
import math
from native_bridge import variance as native_variance_bridge
import torch

def tensors(a):return {k:torch.from_numpy(v.copy()) for k,v in a.items()}

def evaluate(a,P,v,fullcut=None):
    mu=a['mu'];z=torch.abs(mu/torch.sqrt(torch.clamp_min(v,1e-10)))
    count=min(len(v),((int((z<=3.5).sum())+31)//32)*32)
    rows=torch.argsort(z,stable=True)[:count]
    mask=torch.zeros(len(v),dtype=torch.bool);mask[rows]=True
    cut=torch.where(mask,a['fullcut'] if fullcut is None else fullcut,torch.zeros_like(v))
    k3=cut+3*v*a['bpre'];k4=(a['radial']*v)*v
    s=torch.sqrt(torch.clamp_min(v,1e-10));alpha=mu/s
    pdf=torch.exp(-alpha*alpha/2)/math.sqrt(2*math.pi);g=torch.erfc(-alpha/math.sqrt(2))/2
    mg=mu*g+s*pdf
    r2g=(mu*mu+v)*g+mu*s*pdf
    r3g=mu*(mu*mu+3*v)*g+(mu*mu+2*v)*s*pdf
    r4g=(mu**4+6*mu*mu*v+3*v*v)*g+mu*(mu*mu+5*v)*s*pdf
    d=pdf/s;e=-alpha*pdf/v;f=(alpha*alpha-1)*pdf/(v*s)
    m=(mg+k3*e/6)+k4*f/24
    r2=(r2g+k3*d/3)+k4*e/12
    r3=(r3g+k3*g)+k4*d/4
    r4=(r4g+4*k3*mg)+k4*g
    postv=r2-m*m
    post3=(r3-3*m*r2)+2*m**3
    post4=(((r4-4*m*r3)+6*m*m*r2)-3*m**4)-3*postv*postv
    # Actual constant feature columns preserve native FP32 weight-sum results.
    c=a['native_x'].to(torch.float64);zero=torch.zeros_like(v);ones=torch.ones_like(v)
    ls=torch.log(s);ms=m/s;vs=(s*s)*s;vsq=(s*s)*(s*s)
    x=torch.stack((c[:,0],c[:,1],ones,alpha,ls-ls.mean(),g,pdf,mg/s,ms,(m-mg)/s,
        k3/vs,postv/(s*s),k4/vsq,zero,zero,zero,c[:,16],c[:,17],c[:,18],c[:,19],
        post3/vs,post4/vsq,zero,zero,zero,zero,zero,zero,zero,ones*ms.mean()),dim=1).float()
    hx=torch.clamp((x-P['centers'][15])/P['scales'][15],-8,8)
    mask=x.new_tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    hx=hx*mask
    elog=a['elog'].permute(1,0,2).reshape(len(v),48)
    hm=torch.cat((hx[:,5:6]*a['M'],elog),dim=1).clamp(-8,8)
    h=torch.cat((hx,hm),dim=1)
    for name in ('enc0','enc2'):
        h=h@P[name+'_w'].T+P[name+'_b'];h=h/(1+torch.exp(-h))
    state=h@P['enc4_w'].T+P['enc4_b']
    delta=((s.float()*P['rms'][15])*(state@P['head'][15]+P['head_b'][15]))*(1+.25*torch.tanh(P['gain'][15:16]))
    return torch.clamp_min(m+delta.double(),0),x

def push_delta(a,D):
    w=a['W'].double()
    return (w*(D.double()@w)).sum(dim=0)

def star_plane(a,C):
    """Exact native expression order up to the single stored FP32 cast.

    target_minus_cp14 was captured as target-cp_post before trace subtraction.
    Reconstructing from stored FP32 star plus a delta would round too early.
    """
    b=a['b14'].double()
    trace=torch.outer(torch.diagonal(C),b)+C*(2*b)[:,None]
    rest=a['target_minus_cp14'].double()-trace
    return (3*rest.T-2*torch.diag_embed(torch.diagonal(rest))).float()


def star_factor(A,return_columns=False):
    """Current thin-star _lr_factor: selected48, one power step, rank32.

    Selection is exact-forward/piecewise constant in backward. The pure
    Torch matrix products model the algebra, not native rounding order.
    """
    n=A.shape[0]; k=min(48,n); r=min(32,k)
    cn=A.double().square().sum(dim=0)
    columns=torch.sort(torch.argsort(-cn.detach(),stable=True)[:k]).values
    Q=torch.linalg.qr(A[:,columns],mode='reduced')[0]
    Z=torch.linalg.qr(A.T@Q,mode='reduced')[0]
    Q=torch.linalg.qr(A@Z,mode='reduced')[0]
    B=Q.T@A
    U,S,Vh=torch.linalg.svd(B,full_matrices=False)
    result=(Q@U[:,:r],S[:r,None]*Vh[:r])
    return (*result,columns) if return_columns else result


def star_diagonal(a,C):
    A=star_plane(a,C)
    Q,B=star_factor(A)
    wt=a['W'].float().T
    pushed=(wt@Q)@B
    return ((pushed*wt)*wt).sum(dim=1,dtype=torch.float64)


def prepare(a,P):
    """Cache only immutable prefix-dependent baselines for one packet.

    The fit freezes all parent parameters. Clearing/reloading a packet
    invalidates these caches naturally; no cross-network global state.
    """
    if '_physical_base_star' not in a:
        with torch.no_grad():
            a['_physical_base_star']=star_diagonal(a,a['C'].double()).detach()
            a['_physical_base_mean']=evaluate(a,P,a['v'])[0].detach()
    return a


def physical_inputs(a,D):
    if D.shape!=a['C'].shape:
        raise ValueError('post14 covariance shape')
    C=a['C'].double()+D.double()
    fullcut=a['fullcut']+(star_diagonal(a,C)-a['_physical_base_star'])
    v=native_variance_bridge(a,D)
    a['_last_physical_v']=v
    return v,fullcut


def anchored(a,P,D):
    prepare(a,P)
    v,fullcut=physical_inputs(a,D)
    changed,_=evaluate(a,P,v,fullcut)
    return a['prediction'][-1]+(changed-a['_physical_base_mean'])
