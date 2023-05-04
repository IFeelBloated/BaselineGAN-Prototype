import torch.nn as nn
import BaselineGAN.Networks

class Generator(nn.Module):
    def __init__(self, *args, **kw):
        super(Generator, self).__init__()
        
        self.Model = BaselineGAN.Networks.Generator(*args, **kw)
        self.z_dim = kw['NoiseDimension']
        
    def forward(self, x):
        return self.Model(x)
    
class Discriminator(nn.Module):
    def __init__(self, *args, **kw):
        super(Discriminator, self).__init__()
        
        self.Model = BaselineGAN.Networks.Discriminator(*args, **kw)
        
    def forward(self, x):
        return self.Model(x)