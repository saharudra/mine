import torch 
import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os 
import numpy as np 
import matplotlib.pyplot as plt 
from misc.utils import *
import argparse


class Spiral(Dataset):
    def __init__(self, n_points, transform=None):
        super(Spiral, self).__init__()

        self.n_points = n_points
        self.transform = transform
        self.data_x, self.data_y = self.generate_spirals(n_points=int(n_points / 2))

    def generate_spirals(self, n_points, noise=.5):
        """
        Returns the two spirals dataset.
        """
        n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
        d1x = -np.cos(n) * n + np.random.rand(n_points,1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points,1) * noise
        return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
                np.hstack((np.zeros(n_points),np.ones(n_points))))

    def __len__(self):
        """
        :return: Length of the training set
        """
        return self.n_points

    def __getitem__(self, idx):
        """
        :param idx: Data id to be retrieved
        :return: Single sample of spiral data 
        """
        sample = [self.data_x[idx], self.data_y[idx]]
        
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """
    Converts a given list to PyTorch tensor.
    """
    def __call__(self, sample):
        if isinstance(sample, list):
            return torch.Tensor(sample)
        elif isinstance(sample, np.ndarray):
            return torch.from_numpy(sample)
        else:
            print("Input sample is not a list or an ndarray")

 
def spiral_dataloader(params):
    trans=[ToTensor()]

    train_set = Spiral(n_points=params['num_training_points'], transform=None)
    val_set = Spiral(n_points=params['num_val_points'], transform=None)

    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['use_cuda']}

    train_loader = DataLoader(dataset=train_set, batch_size=params['batch_size'], shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=val_set, batch_size=params['batch_size'], shuffle=False, **kwargs)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/spiral_mine.yml', help='Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)

    train_loader, val_loader = spiral_dataloader(params)

    for idx, sample in enumerate(val_loader):
        print(sample)
        print("Only checking the Spiral dataloader")
        import pdb; pdb.set_trace()
    

