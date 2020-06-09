# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 04:45:19 2020

@author: Austin Hsu
"""

from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

@contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)
    
def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= (torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8)
    return d

class VATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=40.0, ip=2):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            #pred = F.softmax(model(x), dim=1)
            pred = F.softmax(model(x).view(3,-1,2), dim=2).view(-1,6)
            
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                #logp_hat = F.softmax(pred_hat, dim=1)
                logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

class VATLoss_onset(nn.Module):

    def __init__(self, xi=1e-6, eps=40.0, ip=2):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss_onset, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            #pred = F.softmax(model(x), dim=1)
            pred = F.softmax(model(x).view(3,-1,2), dim=2).view(-1,6)[:,:2]
            
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                #logp_hat = F.softmax(pred_hat, dim=1)
                logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)[:,:2]
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)[:,:2]
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

class EntropyLoss(nn.Module):
    def __init__(self, entmin_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.entmin_weight = entmin_weight
    def forward(self, softmax_x):
        return -self.entmin_weight * torch.mean(softmax_x * torch.log(softmax_x))  