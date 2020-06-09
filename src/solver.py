# -*- coding: utf-8 -*-
"""
Created on Wed May 27 02:14:00 2020

@author: Austin Hsu
"""

from src.utils.warmup_scheduler import WarmupLR
from src.utils.concat_dataset import ConcatDataset
from src.utils.loss import EntropyLoss, VATLoss, VATLoss_onset
from src.model.PyramidNet_ShakeDrop import PyramidNet_ShakeDrop
from src.model.ResNet18 import ResNet18
import torch
import torch.nn.functional as F
import argparse
import random
from collections import OrderedDict, deque
from pytorch_lightning.core import LightningModule
from pytorch_lightning import _logger as log

class OnOffsetSolver(LightningModule):
    
    def __init__(self, hparams):
        super(OnOffsetSolver, self).__init__()
        
        self.hparams = hparams
            
        # --- Build Model/Loss ---
        self.__build_model(model_type=self.hparams.model_type)
        self.__build_loss()
        
        # --- Meta Data Loader ---
        self.dataset1 = self.hparams.dataset1 # Supervised
        self.dataset2 = self.hparams.dataset2 # Semi-supervised
        self.dataset3 = self.hparams.dataset3 # Instrumental
        self.dataset4 = self.hparams.dataset4 # Validation
        self.dataset5 = self.hparams.dataset5 # Test
        if self.hparams.dataset2 == "None":
            self.dataset2 = None
        if self.hparams.dataset3 == "None":
            self.dataset3 = None
        if self.hparams.dataset4 == "None":
            self.dataset4 = None
        if self.hparams.dataset5 == "None":
            self.dataset5 = None
        self.__metaloader()
        
    def __metaloader(self):
        """Loads meta data"""
        if self.dataset1 == "TONAS":
            self.supervised_dataset = open("./meta/tonas.txt", "r").read().split('\n')
            if self.hparams.val_check_interval > 0:
                raise ValueError("For TONAS training, validation is not done during training.")
        elif self.dataset1 == "DALI":
            self.supervised_dataset = open("./meta/dali_train.txt", "r").read().split('\n')
        else:
            raise NotImplementedError("Given dataset1 name %s is not available, please try from [TONAS, DALI]."%self.dataset1)      
        
        if self.dataset2 is not None:
            if self.dataset2 == "MIR_1K":
                self.semi_supervised_dataset = open("./meta/mir1k.txt", "r").read().split('\n')
            elif self.dataset2 == "Pop_Rhythm":
                self.semi_supervised_dataset = open("./meta/pop_rhythm.txt", "r").read().split('\n')
            else:
                raise NotImplementedError("Given dataset2 name %s is not available, please try from [MIR_1K, Pop_Rhythm]."%self.dataset2)
        else:
            self.semi_supervised_dataset = []
            
        if self.dataset3 is not None:
            if self.dataset3 == "Pop_Rhythm_Instrumental":
                self.instrumental_dataset = open("./meta/pop_rhythm_instrumental.txt", "r").read().split('\n')
            else:
                raise NotImplementedError("Given dataset3 name %s is not available, please try from [Pop_Rhythm_Instrumental]."%self.dataset3)
        else:
            self.instrumental_dataset = [] 
            
        if self.dataset4 is not None:
            if self.dataset4 == "DALI":
                self.valid_dataset = open("./meta/dali_valid.txt", "r").read().split('\n')
            else:
                raise NotImplementedError("Given dataset4 name %s is not available, please try from [DALI]."%self.dataset4)
        else:
            self.valid_dataset = []
        
        if self.dataset5 is not None:
            if self.dataset5 == "DALI":
                self.test_dataset = open("./meta/dali_test.txt", "r").read().split('\n')
            elif self.dataset5 == "ISMIR2014":
                self.test_dataset = open("./meta/ismir_2014.txt", "r").read().split('\n')
            else:
                raise NotImplementedError("Given dataset5 name %s is not available, please try from [DALI, ISMIR2014]."%self.dataset5)
        else:
            self.test_dataset = []
        
    def __build_model(self, model_type):
        if model_type == "Resnet_18":
            self.feature_extractor = ResNet18()
        elif model_type == "PyramidNet_ShakeDrop":
            self.feature_extractor = PyramidNet_ShakeDrop(depth=110, alpha=270, shakedrop=True)
        else:
            raise NotImplementedError("Given model name %s is not available, please try from [Resnet_18, PyramidNet_ShakeDrop]."%model_type)
    
    def __build_loss(self):
        self.enLossFunc = EntropyLoss()
        if "VATo" in self.hparams.loss_type:
            self.smLossFunc = VATLoss()
        else:
            self.smLossFunc = VATLoss_onset()
    
    def forward(self, x):
        return self.feature_extractor(x)
    
    def training_step(self, batch, batch_idx):
        """
        TODO: Verify if use self for VAT or self.feature_extractor
        batch:
            supervised_data (+ instrumental_data)
            semi_supervised_data (optional)
        data:
            feat: (batch_size, 9, 174, 19)
            sdt:  (batch_size, 6)
        """
        
        # --- data collection ---
        if self.dataset2 is None:
            supervised_data = batch
            sup_len = supervised_data[0].size(0)
            feat, sdt = supervised_data
        else:
            supervised_data, semi_supervised_data = batch
            sup_len = supervised_data[0].size(0)
            feat = torch.cat((supervised_data[0], semi_supervised_data[0]), dim=0)
            sdt = torch.cat((supervised_data[1], semi_supervised_data[1]), dim=0)
        
        sdt4 = torch.max(sdt[:,3], sdt[:,5]).view(-1, 1)
        sdt4 = torch.cat((sdt[:,:2], sdt4), dim=1)
        
        sdt_hat = self(feat)
        sdt_hat  = F.softmax(sdt_hat.view(3,-1,2), dim=2).view(-1,6)
        sdt4_hat  = torch.max(sdt_hat[:,3], sdt_hat[:,5]).view(-1,1)
        sdt4_hat = torch.cat((sdt_hat[:,:2], sdt4_hat), dim=1)
        
        # --- supervised loss ---
        super_loss = F.binary_cross_entropy(sdt4_hat[:sup_len], sdt4)
        
        # --- semi-supervised loss ---
        en_loss = 0
        smsup_loss = 0
        if self.dataset2 is not None:
            if 'EntMin' in self.hparams.loss_type:
                # === Entropy Minimization ===
                sdt_u = sdt_hat[sup_len:]      
                en_loss += self.enLossFunc(sdt_u[:, :2])
                en_loss += self.enLossFunc(sdt_u[:,2:4])
                en_loss += self.enLossFunc(sdt_u[:,4: ])
            
            if 'VAT' in self.hparams.loss_type:
                # === VAT Loss ===
                smsup_loss += self.smLossFunc(self, sdt_hat[sup_len:])
        
        # --- Total loss ---
        loss = super_loss + smsup_loss + en_loss
        
        # --- Output ---
        tqdm_dict = {'train_loss': loss, 
                     'supervised_loss': super_loss,
                    }
        if self.dataset2 is not None:
            tqdm_dict['semi-supervised_loss'] = smsup_loss + en_loss
        output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
        })
            
        return output
    
    def validation_step(self, batch, batch_idx):
        # --- data collection ---
        feat, sdt = batch
        
        sdt4 = torch.max(sdt[:,3], sdt[:,5]).view(-1, 1)
        sdt4 = torch.cat((sdt[:,:2], sdt4), dim=1)
        
        sdt_hat = self(feat)
        sdt_hat  = F.softmax(sdt_hat.view(3,-1,2), dim=2).view(-1,6)
        sdt4_hat  = torch.max(sdt_hat[:,3], sdt_hat[:,5]).view(-1,1)
        sdt4_hat = torch.cat((sdt_hat[:,:2], sdt4_hat), dim=1)
        
        # --- supervised loss ---
        loss = F.binary_cross_entropy(sdt4_hat, sdt4)
        
        # --- Output ---
        tqdm_dict = {'val_loss': loss}
        output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
        })
        return output
    
    def validation_epoch_end(self, outputs):
        """
        TODO: Build validation result
        """
        val_loss_mean = 0
        for output in outputs:
            # reduce manually when using dp
            val_loss = output['val_loss']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss    
        
        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result
    
    def test_step(self, batch, batch_idx):
        """
        TODO: Build test step
        NOTE: Should not directly use validation step, since we need to further consider pitch
        """
        return
    
    def test_epoch_end(self, outputs):
        """
        TODO: Build test result
        """
        return
    
    def configure_optimizers(self):
        # --- Optimizer ---
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        
        # --- Scheduler ---
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.max_steps)
        scheduler = WarmupLR(optimizer, self.hparams.lr_warmup, scheduler)
        
        return [optimizer], [scheduler]
    
    def __songloader(self):
        """To choose different songs in every epoch, shuffle again at validation end"""
            
        if self.hparams.shuffle:
            random.shuffle(self.supervised_dataset)
            random.shuffle(self.semi_supervised_dataset)
            random.shuffle(self.instrumental_dataset)
        
        self.song_dict = {
                'supervised': deque([i for i in self.supervised_dataset for _ in range(self.hparams.se)]),
                'semi_supervised': deque([i for i in self.semi_supervised_dataset for _ in range(self.hparams.se)]),
                'instrumental': deque([i for i in self.instrumental_dataset for _ in range(self.hparams.se)]),
                'valid': deque(self.valid_dataset),
                'test': deque(self.test_dataset),
                }
    
    def __dataloader(self, mode):
        """
        self.dataset1: Training dataset
        self.dataset2: semi_supervised dataset
        self.dataset3: Instrumental Dataset
        TODO: Build datasets in src.dataset.py: 
            supervised: TONAS, DALI
            semi_supervised: MIR_1K, Pop_Rhythm
            instrumental: Pop_Rhythm_Instrumental
        TODO: If DALI is used, the second dataset should be loaded with multiple songs
        """
        
        if mode == "train":
            
            # --- Supervised Dataset ---
            supervised_song_name = self.song_dict['supervised'].popleft()
            self.song_dict['supervised'].extend([supervised_song_name])                
            if self.dataset3 is not None:
                instrumental_song_name = self.song_dict['instrumental'].popleft()
                self.song_dict['instrumental'].extend([instrumental_song_name])
            else:
                instrumental_song_name = None
            supervised_song_dataset = ... #(supervised_song_name, instrumental_song_name)
            
            if self.dataset2 is not None:
                semi_supervised_song_name = self.song_dict['semi_supervised'].popleft()
                self.song_dict['semi_supervised'].extend([semi_supervised_song_name])
                semi_supervised_song_dataset = ...
            
            if self.dataset2 is None:
                dataset = supervised_song_dataset
            else:
                dataset = ConcatDataset(supervised_song_dataset, semi_supervised_song_dataset)
            
            return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle,
                                               num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
            
        elif mode == "valid":
            dataset = ...
            return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False,
                                               num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
    
        elif mode == "test":
            dataset = ...
            return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False,
                                               num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
        
    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(mode='train')
    
    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(mode='valid')
    
    def test_dataloader(self):
        log.info('Testing data loader called.')
        return self.__dataloader(mode='test')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument('--model_type', default='FastResnet34', type=str)
        parser.add_argument('--dataset1', default='TONAS', type=str)
        parser.add_argument('--dataset2', default='MIR_1K', type=str)
        parser.add_argument('--dataset3', default='Pop_Rhythm', type=str)
        parser.add_argument('--dataset4', default='None', type=str)
        parser.add_argument('--dataset5', default='ISMIR2014', type=str)
        
        
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--lr_warmup', default=500, type=int)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--max_steps', default=240000, type=int)
        parser.add_argument('--se', default=2, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--shuffle', action='store_true')
        parser.add_argument('--num_workers', default=1, type=int)
        parser.add_argument('--pin_memory', action='store_true')
        parser.set_defaults(
                pin_memory=True,
                shuffle=True)
        return parser