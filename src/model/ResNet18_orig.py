# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:59:16 2020

@author: Austin Hsu
"""

import torchvision.models as models
import torch.nn as nn

class ResNet18(nn.Module):
    
    def __init__(self, feat_num1=9, output_size=6):
        super(ResNet18, self).__init__()
        
        # --- Model ---
        self.resnet18 = models.resnet18(pretrained=False)
        
        # --- Args ---
        self.feat_num1 = feat_num1
        self.output_size = output_size
        self.num_fout = self.resnet18.conv1.out_channels
        self.num_ftrs = self.resnet18.fc.in_features
        
        # --- Switch Model Layers ---
        self.resnet18.conv1 = nn.Conv2d(int(self.feat_num1//3), self.num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.avgpool = nn.AvgPool2d(kernel_size=(17,1), stride=1, padding=0)
        self.resnet18.fc = nn.Linear(self.num_ftrs, self.output_size)
    
    def forward(self, x):
        return self.resnet18(x)