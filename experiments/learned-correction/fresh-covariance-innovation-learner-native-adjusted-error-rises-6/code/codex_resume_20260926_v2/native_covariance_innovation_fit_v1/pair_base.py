import math
import torch

class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left=torch.nn.Linear(94,32,bias=False)
        self.right=torch.nn.Linear(94,32,bias=False)
        self.edge=torch.nn.Linear(3,32,bias=False)
        self.bias=torch.nn.Parameter(torch.zeros(32))
        self.output=torch.nn.Linear(32,1)
        for x in (self.left.weight,self.right.weight):torch.nn.init.normal_(x,0,.15/math.sqrt(94))
        torch.nn.init.normal_(self.edge.weight,0,.15)
        torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)

    def forward(self,F,R,S,s):
        L=self.left(F);B=self.right(F);pieces=[];n=len(F)
        for lo in range(0,n,64):
            hi=min(lo+64,n)
            h=L[lo:hi,None,:]+B[None,:,:]+self.bias
            h=h+R[lo:hi,:,None]*self.edge.weight[:,0]
            h=h+S[lo:hi,:,None]*self.edge.weight[:,1]
            h=h+S[:,lo:hi].T[:,:,None]*self.edge.weight[:,2]
            pieces.append(self.output(torch.tanh(h)).squeeze(-1))
        raw=torch.cat(pieces);D=(raw+raw.T).double()*(.001*torch.outer(s,s))
        return D-torch.diag_embed(torch.diagonal(D))

def inputs(a,norm):
    s=a['s14'];F=(a['F']/norm['feature_scale']).clamp(-8,8)
    R=(a['C']/torch.outer(s,s)).float()/norm['rho_scale']
    S=(a['K']/torch.outer(s*s,s)).float()/norm['third_scale']
    return F,R,S,s
