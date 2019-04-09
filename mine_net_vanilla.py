import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

import numpy as np 
import argparse
import time, datetime
from tqdm import trange
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

from sklearn.metrics import mutual_info_score

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


def sample_batch(data, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:,0].reshape(-1,1),
                                         data[marginal_index][:,1].reshape(-1,1)],
                                       axis=1)
    return batch


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    # unbiasing use moving average
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    # use biased estimator
    # loss = - mi_lb
    
    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et


def train(data, mine_net,mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data,batch_size=batch_size)\
        , sample_batch(data,batch_size=batch_size,sample_mode='marginal')
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        if (i+1)%(log_freq)==0:
            print(result[-1])
    return result


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]



if __name__ == "__main__":
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')

    parser = argparse.ArgumentParser(description='mine')
    parser.add_argument('--config', type=str, default='./configs/mine.yml', 
                        help='Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)
    print(params)

    x = gen_x()
    y = gen_y()

    # sns.scatterplot(x=sample_x[:, 0], y=sample_x[:, 1])
    # sns.scatterplot(x=sample_y[:, 0], y=sample_x[:, 1])

    joint_data = sample_batch(y,batch_size=100,sample_mode='joint')
    # sns.scatterplot(x=joint_data[:,0],y=joint_data[:,1],color='red')
    marginal_data = sample_batch(y,batch_size=100,sample_mode='marginal')
    # sns.scatterplot(x=marginal_data[:,0],y=marginal_data[:,1])
    # plt.show()

    mine_net_indep = StatisticsNetwork(params).cuda()
    mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-3)
    result_indep = train(x,mine_net_indep,mine_net_optim_indep)

    result_indep_ma = ma(result_indep)
    print(result_indep_ma[-1])
    plt.plot(range(len(result_indep_ma)),result_indep_ma)

    if params['use_cuda']:
        mine_net_cor = StatisticsNetwork().cuda()
    mine_net_optim_cor = optim.Adam(mine_net_cor.parameters(), lr=1e-3)
    result_cor = train(y,mine_net_cor,mine_net_optim_cor)


    result_cor_ma = ma(result_cor)
    print(result_cor_ma[-1])
    plt.plot(range(len(result_cor_ma)),result_cor_ma)