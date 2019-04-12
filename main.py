import torch
import torch.nn as nn 

from models.gan import GAN
from trainers.gan import GANTrainerVanilla
from dataloaders.spiral import spiral_dataloader
from misc.utils import *
from misc.logger import Logger

import argparse
import numpy as np 
import time 
import datetime

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')

parser = argparse.ArgumentParser(description='GAN without MI')
parser.add_argument('--config', type=str, default='./configs/spiral_mine.yml',
                        help = 'Path to config file')
opts = parser.parse_args()
params = get_config(opts.config)
print(params)

train_loader, val_loader = spiral_dataloader(params)

model = GAN(params)
if params['use_cuda']:
    model = model.cuda()

logger = Logger(params['logs'])

exp_logs = params['logs'] + params['exp_name'] + '_' + timestamp + '/' 
exp_results = params['results'] + params['exp_name'] + '_' + timestamp + '/'

gan_trainer = GANTrainerVanilla(model, params, train_loader, val_loader, logger, exp_results, exp_logs)

gan_trainer.train()
