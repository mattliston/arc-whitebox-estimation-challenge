"""Differentiable final consumer with current native inputs/messages held fixed."""
import math
import torch

def tensors(a):return {k:torch.from_numpy(v.copy()) for k,v in a.items()}

def evaluate(a,P,v):
    mu=a['mu'];z=torch.abs(mu/torch.sqrt(torch.clamp_min(v,1e-10)))
    count=min(len(v),((int((z<=3.5).sum())+31)//32)*32)
    rows=torch.argsort(z,stable=True)[:count]
    mask=torch.zeros(len(v),dtype=torch.bool);mask[rows]=True
    cut=torch.where(mask,a['fullcut'],torch.zeros_like(v))
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

def anchored(a,P,D):
    baseline,_=evaluate(a,P,a['v'])
    changed,_=evaluate(a,P,a['v']+push_delta(a,D))
    return a['prediction'][-1]+changed-baseline
