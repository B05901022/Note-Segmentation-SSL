# -*- coding: utf-8 -*-
"""
Created on Wed May 27 02:45:57 2020

@author: Austin Hsu
"""

import torch

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)