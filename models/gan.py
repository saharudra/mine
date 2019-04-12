import torch 
import torch.nn as nn 

from models.networks import GeneratorSpiralMine, DiscriminatorSpiralMine

class GAN(nn.Module):
    def __init__(self, params):
        super(GAN, self).__init__()
        self.params = params

        self.gen = GeneratorSpiralMine(self.params)
        self.dis = DiscriminatorSpiralMine(self.params)