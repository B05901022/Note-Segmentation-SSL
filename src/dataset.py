# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:01:13 2020

@author: Austin Hsu
"""

import os
import torch
import numpy as np
import cupy as cp
from src.utils.feature_extraction_cp import full_flow as cp_full_flow
from src.utils.feature_extraction import full_flow as np_full_flow
from src.utils.feature_extraction_cp import test_flow as cp_test_flow
from src.utils.feature_extraction import test_flow as np_test_flow
from src.utils.audio_augment import transform_method

class TrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, dataset1, dataset2, filename1, filename2, mix_ratio,
                 device, use_cp=True, semi=False,
                 num_feat=9, k=9,
                 transform_dict={'cutout'    :False,
                                 'freq_mask' :{'freq_mask_param':100},
                                 'time_mask' :False,
                                 'pitchshift':{'shift_range':48}, 
                                 'addnoise'  :False,
                }):

        # --- Args ---
        self.window_size = 2*k+1
        self.k = k
        self.semi = semi
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.filename1 = filename1
        self.filename2 = filename2
        self.mix_ratio = mix_ratio # mix_ratio: float in range [0., 1.], how many ratio to mix instrumental track into training track
        
        # --- Load File ---
        if use_cp:
            with cp.cuda.Device(device=device.index):
                self.feature = cp_full_flow(
                    os.path.join(data_path, dataset1, 'wav', filename1+'.wav'), 
                    os.path.join(data_path, dataset2, 'wav', filename2+'.wav') if filename2 is not None else None,
                    mix_ratio=self.mix_ratio
                    )
            self.feature = cp.asnumpy(self.feature)
        else:
            self.feature = np_full_flow(
                os.path.join(data_path, dataset1, 'wav', filename1+'.wav'), 
                os.path.join(data_path, dataset2, 'wav', filename2+'.wav') if filename2 is not None else None,
                mix_ratio=self.mix_ratio
                )
        self.feature = torch.from_numpy(self.feature).float()
        self.feature = self.feature.reshape((num_feat,1566//num_feat,-1))
        self.len = self.feature.shape[-1]
        if not self.semi:
            self.sdt = np.load(os.path.join(data_path, dataset1, 'sdt', filename1+'_sdt.npy'))
            self.sdt = torch.from_numpy(self.sdt)

        # --- Pad Length ---
        self.feature = torch.cat([
            torch.zeros((num_feat,1566//num_feat,k)),
            self.feature,
            torch.zeros((num_feat,1566//num_feat,k))
            ], dim=-1)
        
        # --- Transform ---
        self.transform = transform_method(transform_dict)
        self._DataPreprocess()
        
    def __getitem__(self, index):
        frame_feat = self.feature[:, :, index:index+self.window_size]
        frame_feat = self._DataPreprocess(frame_feat)
        if not self.semi:
            frame_sdt = self.sdt[index].float()
            return frame_feat, frame_sdt
        else:
            return frame_feat
    
    def _DataPreprocess(self, feature):
        # --- Normalize (for mask) ---
        self.feature = (self.feature-torch.mean(self.feature))/(torch.std(self.feature)+1e-8)
        
        # --- Augment ---
        # self.feature = self.transform(self.feature.unsqueeze(0)).squeeze(0)
    
    def __len__(self):
        return self.len
    
class EvalDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, dataset1, filename1,
                 device, use_cp=True, no_pitch=False, use_ground_truth=False,
                 num_feat=9, k=9,
                 batch_size=64, num_workers=0, pin_memory=False):
        
        # --- Args ---
        self.window_size = 2*k+1
        self.k = k
        self.dataset = dataset1
        self.filename = filename1
        
        # --- Load File ---
        if use_cp:
            with cp.cuda.Device(device=device.index):
                self.feature, self.pitch = cp_test_flow(os.path.join(data_path, dataset1, 'wav', filename1+'.wav'), use_ground_truth=use_ground_truth, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
            self.feature = cp.asnumpy(self.feature)
            self.pitch = cp.asnumpy(self.pitch)
        else:
            self.feature, self.pitch = np_test_flow(os.path.join(data_path, dataset1, 'wav', filename1+'.wav'), use_ground_ruth=use_ground_truth, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.feature = torch.from_numpy(self.feature).float()
        self.feature = self.feature.reshape((num_feat,1566//num_feat,-1))
        self.len = self.feature.shape[-1]
        self.sdt = np.load(os.path.join(data_path, dataset1, 'sdt', filename1+'_sdt.npy'))
        self.sdt = torch.from_numpy(self.sdt)
        
        if not no_pitch:
            if use_ground_truth:
                self.pitch = np.load(os.path.join(data_path, dataset1, 'pitch', filename1+'_pitch.npy'))
            self.pitch_intervals = np.load(os.path.join(data_path, dataset1, 'pitch_intervals', filename1+'_pi.npy'))
            self.onoffset_intervals = np.load(os.path.join(data_path, dataset1, 'onoffset_intervals', filename1+'_oi.npy'))
            self.onset_intervals = self.onoffset_intervals[:,0]
        self.pitch = self.pitch[:,1]

        # --- Pad Length ---
        self.feature = torch.cat([
            torch.zeros((num_feat,1566//num_feat,k)),
            self.feature,
            torch.zeros((num_feat,1566//num_feat,k))
            ], dim=-1)

        self._DataPreprocess()

    def _DataPreprocess(self, feature):
        # --- Normalize (for mask) ---
        self.feature = (self.feature-torch.mean(self.feature))/(torch.std(self.feature)+1e-8)
        
        # --- Augment ---
        # self.feature = self.transform(self.feature.unsqueeze(0)).squeeze(0)
        
    def __getitem__(self, index):
        frame_feat = self.feature[:, :, index:index+self.window_size]
        frame_feat = self._DataPreprocess(frame_feat)
        frame_sdt = self.sdt[index].float()
        return frame_feat, frame_sdt
    
    def __len__(self):
        return self.len
