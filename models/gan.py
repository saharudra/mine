import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

from models.networks import GeneratorSpiralMine, DiscriminatorSpiralMine

class GANVanilla(nn.Module):
    def __init__(self, params):
        self.gan_params = params['gan']

        self.gen = GeneratorSpiralMine(params['generator'])
        self.dis = DiscriminatorSpiralMine(params['discriminator'])
    
    def forward(self, noise):
        pass
