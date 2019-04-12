import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from models.networks import GeneratorSpiralMine, DiscriminatorSpiralMine

from misc.utils import *
from tqdm import tqdm


class GAN(nn.Module):
    def __init__(self, params):
        super(GAN, self).__init__()
        self.params = params

        self.gen = GeneratorSpiralMine(self.params)
        self.dis = DiscriminatorSpiralMine(self.params)


class GANTrainerVanilla():
    def __init__(self, model, params, train_loader, val_loader, logger):
        super(GANTrainerVanilla, self).__init__()
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        if self.params['use_cuda']:
            self.model = model.cuda()
        else:
            self.model = model

        # Setup the optimizers
        self.gen_opt = torch.optim.Adam(self.model.gen.parameters(), lr=self.params['generator']['lr'])
        self.dis_opt = torch.optim.Adam(self.model.dis.parameters(), lr=self.params['discriminator']['lr'])

    def compute_noise(self):
        noise = torch.randn(self.params['batch_size'], self.params['generator']['z_dim'])
        if self.params['use_cuda']:
            noise = noise.cuda()
        return noise

    def train_dis(self, real_sample):
        noise = self.compute_noise()   
        gen_out = self.model.gen(noise)

        dis_real_score = self.model.dis(real_sample)  # D(x)
        dis_fake_score = self.model.dis(gen_out)  # D(G(z))

        dis_loss = torch.sum(-torch.mean(dis_real_score) + 1e-8) 
                   + torch.log(1 - dis_fake_score + 1e-8)
        
        return dis_loss 

    def train_gen(self):
        noise = self.compute_noise()
        gen_out = self.model.gen(noise)

        dis_fake_score = self.model.dis(gen_out)

        gen_loss = -torch.mean(torch.log(dis_fake_score + 1e-8))
        return gen_loss

    def visualize(self):
        pass

    def train(self):
        iteration = 0
        for epoch in range(self.params['epochs']):
            with trange(len(self.train_loader)) as t:
                self.model.train()
                for idx, sample in enumerate(self.train_loader):
                    loss_dict = {}
                    if self.params['use_cuda']:
                        sample = sample.cuda()

                    # Train discriminator
                    for steps in range(self.params['gan']['d_step']):
                        self.dis_opt.zero_grad()
                        dis_loss = self.train_dis(sample)
                        dis_loss.backward()
                        self.dis_opt.step()
                        # Update loss dict
                        loss_dict = info_dict('dis_loss', (dis_loss.item() / self.params['gan']['d_step']), loss_dict)

                    # Train generator
                    self.gen_opt.zero_grad()
                    gen_loss = self.train_gen()
                    gen_loss.backward()
                    self.gen_opt.step()
                    # Update loss dict
                    loss_dict = info_dict('gen_loss', gen_loss.item(), loss_dict)

                    # Save progress
                    for tag, value in loss_dict.items():
                        self.logger.scalar_summary(tag, value, iteration)
                    iteration += 1

                    # Update progressbar
                    t.set_postfix(loss_dict)
                    t.update()

            if epoch % self.params['loggin_interval'] == 0:
                self.visualize()






        




