# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:27:12 2019
@author: Austin Hsu
"""

import torch
import torch.nn as nn
import math

class ResBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, 
                 stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.branch = nn.Sequential(nn.BatchNorm2d(in_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                                              padding=padding, stride=stride, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channel, out_channel, kernel_size=3, 
                                              padding=1, stride=1, bias=False),
                                    nn.BatchNorm2d(out_channel))
        self.downsample = stride != 1 or in_channel != out_channel
        if self.downsample:
            self.shortcut   = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1,
                                                      stride=stride, bias=False),
                                            nn.BatchNorm2d(out_channel))
        else: 
            self.shortcut = None
                
    def forward(self, x):
        h0 = self.shortcut(x) if self.downsample else x
        h  = self.branch(x)
        return h+h0        
    
class ResNet18(nn.Module):
    
    def __init__(self, conv1_in_channel=9, depth=18, num_class=6, block=ResBlock):
        super(ResNet18, self).__init__()
        
        n_units = (depth - 2) // 8
        self.in_chs  = [64] + [2**(6+i//2) for i in range (n_units * 4)]
        #print('in_channel_list:',self.in_chs)
        
        # Stochastic Depth
        self.p_L = 0.5
        linear_decay = lambda x: (1-self.p_L) * x / (4*n_units)
        
        self.u_idx   = 0
        
        ### Model ###
        
        # input shape (batch, 3, 522, 19)
        
        self.conv1   = nn.Conv2d(conv1_in_channel, self.in_chs[0], kernel_size=(7,7),
                               stride=(2,2), padding=(3,3), bias=False)
        self.bn1     = nn.BatchNorm2d(self.in_chs[0])
        self.relu1   = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1  = self._make_layer(n_units, block, 1, (1,1))
        self.layer2  = self._make_layer(n_units, block, 2, (1,1))
        self.layer3  = self._make_layer(n_units, block, 2, (1,1))
        self.layer4  = self._make_layer(n_units, block, 2, (1,1))
        
        self.bn_out  = nn.BatchNorm2d(self.in_chs[-1])
        self.relu_out= nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=(6,1), stride=1, padding=0)
        #self.avgpool2= nn.AvgPool2d(kernel_size=(33,2), stride=1, padding=0)
        self.fc_out  = nn.Linear(self.in_chs[-1], num_class) #nn.Linear(self.in_chs[-1]*34, num_class)
        
        #output shape (batch, 6)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        #print('1',x.shape)
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.maxpool(h)
        #print('2',h.shape)
        h = self.layer1(h)
        #print('3',h.shape)
        #print('==================================================')
        h = self.layer2(h)
        #print('4',h.shape)
        #print('==================================================')
        h = self.layer3(h)
        #print('5',h.shape)
        #print('==================================================')
        h = self.layer4(h)
        #print('6',h.shape)
        #print('==================================================')
        h = self.relu_out(self.bn_out(h))
        #print('7',h.shape)
        h = self.avgpool(h)
        #print('8',h.shape)
        h = h.view(h.size(0), -1)
        #print('9',h.shape)
        h = self.fc_out(h)
        return h
    
    def _make_layer(self, n_units, block, stride=1, padding=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(block(self.in_chs[self.u_idx], self.in_chs[self.u_idx+1], stride, padding))
            self.u_idx += 1
            stride = 1
            padding = 1
        return nn.Sequential(*layers)