import math
import torch

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

    def forward(self,F,R,S,G,s):
        L=self.left(F);B=self.right(F);pieces=[];n=len(F)
        for lo in range(0,n,64):
            hi=min(lo+64,n)
            h=L[lo:hi,None,:]+B[None,:,:]+self.bias
            h=h+R[lo:hi,:,None]*self.edge.weight[:,0]
            h=h+S[lo:hi,:,None]*self.edge.weight[:,1]
            h=h+S[:,lo:hi].T[:,:,None]*self.edge.weight[:,2]
            h=h+G[lo:hi,:,None]*self.edge.weight[:,3]
            pieces.append(self.output(torch.tanh(h)).squeeze(-1))
        raw=torch.cat(pieces);D=(raw+raw.T).double()*(.001*torch.outer(s,s))
        return D-torch.diag_embed(torch.diagonal(D))

