import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 
import argparse
import time, datetime
from tqdm import trange
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

from misc.logger import Logger
from misc.utils import *


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


def gen_x():
    return np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=300)


def gen_y():
    return np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=300)

if __name__ == "__main__":
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')

    parser = argparse.ArgumentParser(description='mine')
    parser.add_argument('--config', type=str, default='./configs/mine.yml', 
                        help='Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)
    print(params)

    sample_x = gen_x()
    sample_y = gen_y()

    sns.scatterplot(x=sample_x[:, 0], y=sample_x[:, 1])
    sns.scatterplot(x=sample_y[:, 0], y=sample_x[:, 1])

