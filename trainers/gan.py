import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from misc.utils import *
from tqdm import trange
import matplotlib.pyplot as plt 
import numpy as np


class GANTrainerVanilla():
    def __init__(self, model, params, train_loader, val_loader, logger, exp_results, exp_logs):
        super(GANTrainerVanilla, self).__init__()
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.exp_results = exp_results
        self.exp_logs = exp_logs

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

        dis_loss = torch.sum(-torch.mean(torch.log(dis_real_score + 1e-8) + torch.log(1 - dis_fake_score + 1e-8)))
        
        return dis_loss 

    def train_gen(self):
        noise = self.compute_noise()
        gen_out = self.model.gen(noise)

        dis_fake_score = self.model.dis(gen_out)

        gen_loss = -torch.mean(torch.log(dis_fake_score + 1e-8))
        return gen_loss

    def visualize(self, epoch):
        """
        We will be generating 1000 samples from the generator, equivalent to the number
        of samples in the val_loader and save the overlayed plot.
        """
        self.model.eval()  # Set model to eval mode

        # Get validation sample, no need to put on cuda as this is only being visualized
        val_samples = next(iter(self.val_loader))
        val_samples = val_samples.numpy()
        val_gen_samples = []
        for i in range(int(self.params['num_val_points'] / self.params['batch_size'])):
            val_noise = self.compute_noise()
            val_gen_out = self.model.gen(val_noise).detach().cpu().numpy()
            val_gen_samples.append(val_gen_out)
        val_gen_samples = np.vstack(val_gen_samples)
        
        # Plot validation and generated samples
        plt.title('GAN w/o MI')
        plt.scatter(val_samples[:, 0], val_samples[:, 1], marker='.', label='original', color='green')
        plt.scatter(val_gen_samples[:, 0], val_gen_samples[:, 1], marker='.', label='generated', color='blue')
        plt.legend(loc='lower left')
        img_filename = self.exp_results + 'epoch_' + str(epoch) + '_spiral.jpg'
        plt.savefig(img_filename)
        plt.close()

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
                self.visualize(epoch)


class GANTrainerMI():
    def __init__(self, model, params, train_loader, val_loader, logger, exp_results, exp_logs):
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.exp_results = exp_results
        self.exp_logs = exp_logs

        if self.params['use_cuda']:
            self.model = model.cuda()
        else:
            self.model = model

        # Setup the optimizers
        self.gen_opt = torch.optim.Adam(self.model.gen.parameters(), lr=self.params['generator']['lr'])
        self.dis_opt = torch.optim.Adam(self.model.dis.parameters(), lr=self.params['discriminator']['lr'])
        self.mi_opt = torch.optim.Adam(self.model.mi.parameters(), lr=self.params['mi']['lr']) 

    def compute_noise(self):
        noise = torch.randn(self.params['batch_size'], self.params['generator']['z_dim'])
        if self.params['use_cuda']:
            noise = noise.cuda()
        return noise

    def train_dis(self):
        pass
    
    def train_gen(self):
        pass 

    def train_mine(self):
        pass

    def train(self):
        pass

    def visualize(self):
        pass