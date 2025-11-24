import time
import numpy as np
import torch
from torch import nn

import parameters


sigma = parameters.sigma
device = parameters.device
n_fourier_features = parameters.n_fourier_features

print('sigma: '+str(sigma)+'')
print('_______________________')
print('device: '+str(device)+'')
print('_______________________')
print('n_fourier_features: '+str(n_fourier_features)+'')
print('_______________________')



class FourierFeatures(torch.nn.Module):
    def __init__(self, n_input, n_fourier_features):
        
        super(FourierFeatures, self).__init__()
        
        self.device = device
        self.sigma = sigma
        self.n_input = 2
        self.n_output = n_fourier_features
        
        self.B = torch.nn.Parameter(torch.normal(0, self.sigma, size=(self.n_input, self.n_output)), requires_grad=False).to(self.device)
        
        
    def forward(self, x):
            
        return torch.cat([torch.sin(torch.matmul(x, self.B)), torch.cos(torch.matmul(x, self.B))], dim=-1)

class simple_NN(nn.Module):
    
    def __init__(self):
        super(simple_NN, self).__init__()
        
        self.fourier_features = FourierFeatures(n_input = 2, n_fourier_features = n_fourier_features)
        #self.n_fourier_features = 4
        
        self.NN = nn.Sequential(
            nn.Linear(2*n_fourier_features+2, 256),
            SinActivation(),
            nn.Linear(256, 256),
            SinActivation(),
            nn.Linear(256, 128),
            SinActivation(),
            nn.Linear(128, 128),
            SinActivation(),
            nn.Linear(128, 64),
            SinActivation(),
            nn.Linear(64, 64),
            SinActivation(),
            nn.Linear(64, 32),
            SinActivation(),
            nn.Linear(32, 1),
            )
       
  

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)  # junta os inputs

        xy_FF = self.fourier_features(xy)
        
        xy = torch.cat([xy, xy_FF], dim=1)

        out = self.NN(xy)

        return out
        

        
        
class SinActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)


   
    

    
    
