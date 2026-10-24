import math
import torch
from projection import project


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left=torch.nn.Linear(94,32,bias=False)
        self.right=torch.nn.Linear(94,32,bias=False)
        self.edge=torch.nn.Linear(4,32,bias=False)
        self.bias=torch.nn.Parameter(torch.zeros(32))
        self.output=torch.nn.Linear(32,1)
        for x in (self.left.weight,self.right.weight):torch.nn.init.normal_(x,0,.15/math.sqrt(94))
        torch.nn.init.normal_(self.edge.weight,0,.15)
        torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)

    def forward(self,F,R,S,s,U):
        L=self.left(F);B=self.right(F);pieces=[];n=len(F)
        memory=(U/(.002*torch.outer(s,s))).to(F.dtype)
        for lo in range(0,n,64):
            hi=min(lo+64,n)
            h=L[lo:hi,None,:]+B[None,:,:]+self.bias
            h=h+R[lo:hi,:,None]*self.edge.weight[:,0]
            h=h+S[lo:hi,:,None]*self.edge.weight[:,1]
            h=h+S[:,lo:hi].T[:,:,None]*self.edge.weight[:,2]
            h=h+memory[lo:hi,:,None]*self.edge.weight[:,3]
            pieces.append(self.output(torch.tanh(h)).squeeze(-1))
        raw=torch.cat(pieces)
        return (raw+raw.T).double()*(.001*torch.outer(s,s))


def inputs(a,norm):
    s=a['sigma'];F=(a['F']/norm['feature_scale']).clamp(-8,8)
    R=(a['C']/torch.outer(s,s)).to(F.dtype)/norm['rho_scale']
    S=(a['K']/torch.outer(s*s,s)).to(F.dtype)/norm['third_scale']
    return F,R,S,s


def rollout(model,cells,terminal,norm,mode):
    assert mode in ('history','reset')
    D=torch.zeros_like(cells[0]['C']);e=torch.zeros_like(cells[0]['m']);states=[]
    for index,a in enumerate(cells):
        if index:
            W=a['W'].double();U=W.T@D@W
            e=a['gate']*(W.T@e)+a['pdf']/(2*a['sigma'])*torch.diagonal(U)
        else:U=torch.zeros_like(D)
        if mode=='history':
            carry=torch.outer(a['gate'],a['gate'])*U;memory=U
        else:carry=torch.zeros_like(U);memory=torch.zeros_like(U)
        D=project(carry+model(*inputs(a,norm),memory),a['m'])
        states.append(D)
    W=terminal['W'].double();U=W.T@D@W
    e=terminal['gate']*(W.T@e)+terminal['pdf']/(2*terminal['sigma'])*torch.diagonal(U)
    forecast=(terminal['parent']+e).clamp_min(0)
    return states,e,forecast
