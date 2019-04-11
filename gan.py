import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from models.networks import GeneratorSpiralMine, DiscriminatorSpiralMine

from misc.utils import *
from tqdm import tqdm


class GANTrainerVanilla(nn.Module):
    def __init__(self, params):
        super(GANTrainerVanilla, self).__init__()
        self.params = params

        # Initiate the networks
        self.gen = GeneratorSpiralMine(self.params)
        self.dis = DiscriminatorSpiralMine(self.params)

        # Setup the optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.params['generator']['lr'])
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.params['discriminator']['lr'])

    def compute_noise(self):
        noise = torch.randn(self.params['batch_size'], self.params['generator']['z_dim'])
        if self.params['use_cuda']:
            noise = noise.cuda()
        return noise

    def train_dis(self, real_sample):
        noise = self.compute_noise()   
        gen_out = self.gen(noise)

        dis_real_score = self.dis(real_sample)  # D(x)
        dis_fake_score = self.dis(gen_out)  # D(G(z))

        dis_loss = torch.sum(-torch.mean(dis_real_score) + 1e-8) 
                   + torch.log(1 - dis_fake_score + 1e-8)
        
        return dis_loss 

    def train_gen(self):
        noise = self.compute_noise()
        gen_out = self.gen(noise)

        dis_fake_score = self.dis(gen_out)

        gen_loss = -torch.mean(torch.log(dis_fake_score + 1e-8))
        return gen_loss

    def train(self):
        




