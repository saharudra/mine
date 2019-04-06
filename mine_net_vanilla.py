import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
import argparse
import time, datetime
from tqdm import trange

from misc.logger import Logger


class StatisticsNetwork(nn.Module):
    def __init__(self, params):
        super(StatisticsNetwork, self).__init__()
        self.statistics_network_params = params['statistics_network']

        self.network_definition()

    def network_definition(self):
        self.statistics_network = nn.Sequential(
            nn.Linear(self.statistics_network_params['nc'], self.statistics_network_params['hidden']),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.statistics_network_params['hidden'], self.statistics_network_params['hidden']),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.statistics_network_params['hidden'], self.statistics_network_params['od'])
        )

    def forward(self, x):
        T = self.statistics_network(x)
        return T


if __name__ == __main__:
    T_net = StatisticsNetwork()
