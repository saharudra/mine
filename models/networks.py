import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Mine(nn.Module):
    def __init__(self, params):
        super(Mine, self).__init__()
        self.params = params['mine']
        self.fc1_x = nn.Linear(self.params['var1_size'], self.params['hidden'], bias=False)
        self.fc1_y = nn.Linear(self.params['var2_size'], self.params['hidden'], bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(self.params['hidden']))
        self.fc2 = nn.Linear(self.params['hidden'], self.params['hidden'])
        self.fc3 = nn.Linear(self.params['hidden'], self.params['oc'])

        self.ma_et = None

    def forward(self, x, y):
        x = self.fc1_x(x)
        y = self.fc1_y(y)

        out = F.leaky_relu(x + y + self.fc1_bias, negative_slope=2e-1)
        out = F.leaky_relu(self.fc2(out), negative_slope=2e-1)
        out = self.fc3(out)
        return out


class GeneratorSpiralMine(nn.Module):
    def __init__(self, params):
        super(GeneratorSpiralMine, self).__init__()
        self.gparams = params['generator']

        self.generator_network()

    def generator_network(self):
        self.generator = nn.Sequential(
            nn.Linear(self.gparams['z_dim'], self.gparams['hidden']),
            nn.BatchNorm1d(self.gparams['hidden']),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.gparams['hidden'], self.gparams['oc']),
        )

    def forward(self, noise):
        # Expecting the dimensions of the noise vector to be [bs, z_dim]
        gen_sample = self.generator(noise)
        return gen_sample


class DiscriminatorSpiralMine(nn.Module):
    def __init__(self, params):
        super(DiscriminatorSpiralMine, self).__init__()
        self.dparams = params['discriminator']

        self.discriminator_network()

    def discriminator_network(self):
        self.discriminator = nn.Sequential(
            nn.Linear(self.dparams['nc'], self.dparams['hidden']),
            nn.BatchNorm1d(self.dparams['hidden']),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.dparams['hidden'], self.dparams['hidden']),
            nn.BatchNorm1d(self.dparams['hidden']),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.dparams['hidden'], self.dparams['oc']),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out
