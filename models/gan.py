import torch 
import torch.nn as nn 

from models.networks import GeneratorSpiralMine, DiscriminatorSpiralMine, Mine

class GAN(nn.Module):
    def __init__(self, params):
        super(GAN, self).__init__()
        self.params = params

        self.gen = GeneratorSpiralMine(self.params)
        self.dis = DiscriminatorSpiralMine(self.params)


class GAN_MI(nn.Module):
    def __init__(self, params):
        super(GAN_MI, self).__init__()
        self.params = params

        self.gen = GeneratorSpiralMine(self.params)
        self.dis = DiscriminatorSpiralMine(self.params)
        self.mi = Mine(self.params)
