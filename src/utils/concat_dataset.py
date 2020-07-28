# -*- coding: utf-8 -*-
"""
Created on Wed May 27 02:45:57 2020

@author: Austin Hsu
"""

import torch

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.len_list = [len(d) for d in self.datasets]

    def __getitem__(self, i):
        return tuple(self.datasets[d][i%self.len_list[d]] for d in range(len(self.datasets))) #tuple(d[i] for d in self.datasets)

    def __len__(self):
        return self.len_list[0] #min(len(d) for d in self.datasets)