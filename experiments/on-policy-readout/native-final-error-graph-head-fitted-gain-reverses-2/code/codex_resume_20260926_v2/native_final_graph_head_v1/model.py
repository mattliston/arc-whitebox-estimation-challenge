import torch

class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(151,512),torch.nn.SiLU(),
                                        torch.nn.Linear(512,256),torch.nn.SiLU(),
                                        torch.nn.Linear(256,1))
        torch.nn.init.zeros_(self.layers[4].weight)
        torch.nn.init.zeros_(self.layers[4].bias)

    def forward(self,x):
        return self.layers(x).squeeze(-1)
