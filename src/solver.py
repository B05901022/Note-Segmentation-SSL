# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:20:14 2020

@author: Austin Hsu
"""

from src.dataset import TrainDataset, EvalDataset
from src.utils.warmup_scheduler import WarmupLR
from src.utils.concat_dataset import ConcatDataset
from src.utils.loss import EntropyLoss, VATLoss, VATLoss_onset
from src.utils.evaluate_tools import Smooth_sdt6_modified, Naive_pitch #, freq2pitch
from src.model.PyramidNet_ShakeDrop import PyramidNet_ShakeDrop
from src.model.ResNet18 import ResNet18
from src.utils.audio_augment import transform_method
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import mir_eval
import apex.amp as amp
import numpy as np
import argparse
from collections import OrderedDict, deque

class OnOffsetSolver:
    """
    Args:
        model_type: Model used
        loss_type: Loss used
        dataset1: Supervised dataset
        dataset2: Semi-supervised dataset
        dataset3: Instrumental dataset
        dataset4: Validation dataset
        dataset5: Test dataset
        mix_ratio: Ratio of instrumental dataset to mix with supervised dataset
        meta_path: Datapath to meta files
        data_path: Datapath to wav files
        lr: Learning rate
        lr_warmup: Warmup steps for learning rate
        max_steps: Max training steps
        se: Re-train same sample for how many epochs
        num_feat: Numbers of channel dimension
        k: Window side from one side (Window size=2*k+1)
        shuffle: Shuffle dataset or not
        batch_size: Batch size
        num_workers: Num workers
        pin_memory: Pin memory
        use_cp: Use CuPy instead of Numpy for data preprocessing        
    """
    def __init__(self, hparams):
        
        self.hparams = hparams
        self.hparams.use_amp = False
        self.best_epoch = 0
        self.test_no_offset = False

        self.check_cp_available = lambda x: True #("DALI" not in x) and ("MedleyDB" not in x) and ("CMedia" not in x)
        
        # --- Build Model/Loss ---
        self.__build_model(model_type=self.hparams.model_type)
        self.__build_loss()
        
        # --- Meta Files ---
        self.meta_dict = {
            'TONAS': "tonas.txt",
            'DALI_train': "dali_train.txt",
            "DALI_valid": "dali_valid.txt",
            "DALI_test": "dali_test.txt",
            "DALI_orig_train": "dali_orig_train.txt",
            "DALI_orig_valid": "dali_orig_valid.txt",
            "DALI_orig_test": "dali_orig_test.txt",
            "DALI_demucs_train": "dali_demucs_train.txt",
            "DALI_demucs_valid": "dali_demucs_valid.txt",
            "DALI_demucs_test": "dali_demucs_test.txt",
            "CMedia": "cmedia.txt",
            "CMedia_demucs": "cmedia_demucs.txt",
            "MIR_1K": "mir1k.txt",
            "MIR_1K_Polyphonic": "mir1k_polyphonic.txt",
            "MIR_1K_Instrumental": "mir1k_instrumental.txt",
            "Pop_Rhythm": "pop_rhythm.txt",
            "Pop_Rhythm_Instrumental": "pop_rhythm_instrumental.txt",
            "ISMIR2014": "ismir_2014.txt",
            "MedleyDB": "medleydb.txt",
        }

        # --- Available Datasets ---
        self.available_train_dataset = ["TONAS", "DALI_train", "DALI_orig_train", "DALI_demucs_train", "CMedia", "CMedia_demucs"]
        self.available_semi_dataset = ["MIR_1K", "MIR_1K_Polyphonic", "Pop_Rhythm", "DALI_train", "DALI_orig_train", "DALI_demucs_train", "MedleyDB", "CMedia", "CMedia_demucs"]
        self.available_inst_dataset = ["Pop_Rhythm_Instrumental", "MIR_1K_Instrumental"]
        self.available_valid_dataset = ["DALI_valid", "DALI_orig_valid", "DALI_demucs_valid"]
        self.available_test_dataset = ["DALI_test", "DALI_orig_test", "DALI_demucs_test", "ISMIR2014", "CMedia", "CMedia_demucs"]

        # --- Meta Data Loader ---
        self.dataset1 = self.hparams.dataset1.split("|") # Supervised
        self.dataset2 = self.hparams.dataset2.split("|") # Semi-supervised
        self.dataset3 = self.hparams.dataset3.split("|") # Instrumental
        self.dataset4 = self.hparams.dataset4.split("|") # Validation
        self.dataset5 = self.hparams.dataset5.split("|") # Test
        if self.hparams.dataset2 == "None":
            self.dataset2 = []
        if self.hparams.dataset3 == "None":
            self.dataset3 = []
        if self.hparams.dataset4 == "None":
            self.dataset4 = []
        if self.hparams.dataset5 == "None":
            self.dataset5 = []
        print(f"Training dataset: {self.dataset1}")
        print(f"Semi-supervised dataset: {self.dataset2}")
        print(f"Instrumental dataset: {self.dataset3}")
        print(f"Validation dataset: {self.dataset4}")
        print(f"Testing dataset: {self.dataset5}")
        print()
        self.__metaloader()
        
    def __metaloader(self):
        """Loads meta data"""
        # === TRAIN DATASET ===
        self.supervised_datalist = []
        for train_dataset in self.dataset1:
            if train_dataset in self.available_train_dataset:
                self.supervised_datalist += [(i, train_dataset) for i in open(os.path.join(self.hparams.meta_path, self.meta_dict[train_dataset]), "r").read().split('\n')]
            else:
                raise NotImplementedError(f"Given dataset1 name {train_dataset} is not available, please try from {self.available_train_dataset}.")      
        
        # === SEMI-SUPERVISED DATASET ===
        self.semi_supervised_datalist = []
        for semi_dataset in self.dataset2:
            if semi_dataset in self.available_semi_dataset:
                self.semi_supervised_datalist += [(i, semi_dataset) for i in open(os.path.join(self.hparams.meta_path, self.meta_dict[train_dataset]), "r").read().split('\n')]
            else:
                raise NotImplementedError(f"Given dataset2 name {semi_dataset} is not available, please try from {self.available_semi_dataset}.")
        
        # === INSTRUMENTAL DATASET ===
        self.instrumental_datalist = []
        for inst_dataset in self.dataset3:
            if inst_dataset in self.available_inst_dataset:
                self.instrumental_datalist += [(i, inst_dataset) for i in open(os.path.join(self.hparams.meta_path, self.meta_dict[inst_dataset]), "r").read().split('\n')]
            else:
                raise NotImplementedError(f"Given dataset3 name {inst_dataset} is not available, please try from {self.available_inst_dataset}.")
        
        # === VALID DATASET ===
        self.valid_datalist = []
        for valid_dataset in self.dataset4:
            if valid_dataset in self.available_valid_dataset:
                self.valid_datalist += [(i, valid_dataset) for i in open(os.path.join(self.hparams.meta_path, self.meta_dict[valid_dataset]), "r").read().split('\n')]
            else:
                raise NotImplementedError(f"Given dataset4 name {valid_dataset} is not available, please try from {self.available_valid_dataset}.")
        
        # === TEST DATASET ===
        self.test_datalist = []
        for test_dataset in self.dataset5:
            if test_dataset in self.available_test_dataset:
                self.test_datalist += [(i, test_dataset) for i in open(os.path.join(self.hparams.meta_path, self.meta_dict[test_dataset]), "r").read().split('\n')]
            else:
                raise NotImplementedError(f"Given dataset5 name {test_dataset} is not available, please try from {self.available_test_dataset}.")
    
    def __build_model(self, model_type):
        if model_type == "Resnet_18":
            self.feature_extractor = ResNet18()
        elif model_type == "PyramidNet_ShakeDrop":
            self.feature_extractor = PyramidNet_ShakeDrop(depth=110, alpha=270, shakedrop=True)
        else:
            raise NotImplementedError("Given model name %s is not available, please try from [Resnet_18, PyramidNet_ShakeDrop]."%model_type)
            
    def __build_loss(self):
        self.trLossFunc = nn.BCELoss()
        self.enLossFunc = EntropyLoss()
        if "VATo" in self.hparams.loss_type:
            self.smLossFunc = VATLoss_onset()
        else:
            self.smLossFunc = VATLoss()
    
    def forward(self, x):
        return self.feature_extractor(x)
    
    def training_step(self, batch, batch_idx):
        """
        batch:
            supervised_data (+ instrumental_data): feat, sdt
            semi_supervised_data (optional): feat
        data:
            feat: (batch_size, 9, 174, 19)
            sdt:  (batch_size, 6)
        """
        
        # --- data collection ---
        if self.dataset2 is []:
            supervised_data = batch
            sup_len = supervised_data[0].size(0)
            feat, sdt = supervised_data
        else:
            supervised_data, semi_supervised_data = batch
            sup_len = supervised_data[0].size(0)
            if 'EntMin' in self.hparams.loss_type:
                feat = torch.cat((supervised_data[0], semi_supervised_data), dim=0)
            else:
                feat = supervised_data[0]
            sdt = supervised_data[1]
        
        sdt4 = torch.max(sdt[:,3], sdt[:,5]).view(-1, 1)
        sdt4 = torch.cat((sdt[:,:2], sdt4), dim=1)
        
        sdt_hat = self.forward(feat)
        sdt_hat  = F.softmax(sdt_hat.view(3,-1,2), dim=2).view(-1,6)
        sdt4_hat  = torch.max(sdt_hat[:,3], sdt_hat[:,5]).view(-1,1)
        sdt4_hat = torch.cat((sdt_hat[:,:2], sdt4_hat), dim=1)
        
        # --- supervised loss ---
        super_loss = self.trLossFunc(sdt_hat[:sup_len], sdt)*6 + self.trLossFunc(sdt4_hat[:sup_len], sdt4)*3
        
        # --- semi-supervised loss ---
        en_loss = 0
        smsup_loss = 0
        if self.dataset2 != []:
            if 'EntMin' in self.hparams.loss_type:
                # === Entropy Minimization ===
                sdt_u = sdt_hat[sup_len:]      
                en_loss += self.enLossFunc(sdt_u[:, :2])
                en_loss += self.enLossFunc(sdt_u[:,2:4])
                en_loss += self.enLossFunc(sdt_u[:,4: ])
            
            if 'VAT' in self.hparams.loss_type:
                # === VAT Loss ===
                smsup_loss += self.smLossFunc(self.feature_extractor, semi_supervised_data)
        
        # --- Total loss ---
        loss = super_loss + smsup_loss + en_loss
        
        # --- Output ---
        tqdm_dict = {
            'train_loss': loss.item(), 
            'supervised_loss': super_loss.item(),
        }
        if self.dataset2 != []:
            tqdm_dict['semi-supervised_loss'] = (smsup_loss + en_loss).item()
        output = OrderedDict({
                'loss': loss,
                'progress_bar': {i:round(tqdm_dict[i],8) for i in tqdm_dict.keys()},
                'log': tqdm_dict
        })
            
        return output
    
    def validation_step(self, batch, batch_idx):
        # --- data collection ---
        feat, sdt = batch
        
        sdt4 = torch.max(sdt[:,3], sdt[:,5]).view(-1, 1)
        sdt4 = torch.cat((sdt[:,:2], sdt4), dim=1)
        
        sdt_hat = self.forward(feat)
        sdt_hat  = F.softmax(sdt_hat.view(3,-1,2), dim=2).view(-1,6)
        sdt4_hat  = torch.max(sdt_hat[:,3], sdt_hat[:,5]).view(-1,1)
        sdt4_hat = torch.cat((sdt_hat[:,:2], sdt4_hat), dim=1)
        
        # --- supervised loss ---
        loss = self.trLossFunc(sdt_hat, sdt)*6 + self.trLossFunc(sdt4_hat, sdt4)*3
        
        # --- Output ---
        tqdm_dict = {'val_loss': loss.item()}
        output = OrderedDict({
                'loss': loss.item(),
                'progress_bar': {i:round(tqdm_dict[i],8) for i in tqdm_dict.keys()},
                'log': tqdm_dict
        })
        return output
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        for output in outputs:
            val_loss = output['val_loss']
            val_loss_mean += val_loss    
        
        val_loss_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {'val_loss': val_loss_mean, 'progress_bar': tqdm_dict, 'log': tqdm_dict}
        return result
    
    def test_step(self, batch, batch_idx):
        # --- data collection ---
        feat, sdt = batch
        
        sdt_hat = self.forward(feat)
        sdt_hat  = F.softmax(sdt_hat.view(3,-1,2), dim=2).view(-1,6)
        
        # --- Output ---
        output = OrderedDict({
                'sdt': sdt_hat.detach().cpu(),
        })
        return output
    
    def test_epoch_end(self, outputs):
        # --- data collection ---
        p_np = self.testing_dataset.pitch
        freq_ans = self.testing_dataset.pitch_intervals
        ans_np = self.testing_dataset.onoffset_intervals
        onset_ans_np = self.testing_dataset.onset_intervals
        predict_on_notes_np = outputs.numpy()
        
        # --- evaluation ---
        pitch_intervals, sSeq_np, dSeq_np, onSeq_np, offSeq_np, conflict_ratio = Smooth_sdt6_modified(predict_on_notes_np, threshold=0.5) # list of onset secs, ndarray
        F_on, P_on, R_on = mir_eval.onset.f_measure(onset_ans_np, pitch_intervals[:,0], window=0.05)
        offset_ratio = None if self.test_no_offset else 0.2
        #try:
        freq_est = Naive_pitch(p_np, pitch_intervals)

        (P, R, F1) = mir_eval.transcription.offset_precision_recall_f1(ans_np, pitch_intervals, offset_ratio=0.2, offset_min_tolerance=0.05)
        P_p, R_p, F_p, AOR = mir_eval.transcription.precision_recall_f1_overlap(ans_np, freq_ans, pitch_intervals, freq_est, pitch_tolerance=50.0, offset_ratio=offset_ratio)
            #P_p, R_p, F_p, AOR = mir_eval.transcription.precision_recall_f1_overlap(ans_np, freq_ans, pitch_intervals, freq_est, pitch_tolerance=50.0)    
        #except:
        #    P, R, F1, P_p, R_p, F_p, AOR = 0,0,0,0,0,0,0
        
        tqdm_dict = {
            'Onset_F1': F_on,
            'Offset_F1': F1,
            'Transcription_F1': F_p,
            'Onset_Precision': P_on,
            'Offset_Precision': P,
            'Transcription_Precision': P_p,
            'Onset_Recall': R_on,
            'Offset_Recall': R,
            'Transcription_Recall': R_p,
            'Average_Overlap_Ratio': AOR,
            'Conflict_Ratio': conflict_ratio,
        }
        
        output = OrderedDict({
            'File': self.testing_dataset.filename,
            'log': tqdm_dict
        })
        
        return output
    
    def configure_optimizers(self):
        # --- Optimizer ---
        optimizer = torch.optim.AdamW(self.feature_extractor.parameters(), lr=self.hparams.lr)
        
        # --- Scheduler ---
        scheduler = None
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.max_steps)
        #scheduler = WarmupLR(optimizer, self.hparams.lr_warmup, scheduler)
        
        return optimizer, scheduler
    
    def initsonglist(self):
        """
        TODO: Call self.__initsonglist() at validation_epoch_end and test_epoch_end if song list not ended.
        """
        """To choose different songs in every epoch, shuffle again at validation end"""
        if self.hparams.shuffle:
            random.shuffle(self.supervised_datalist)
            random.shuffle(self.semi_supervised_datalist)
            random.shuffle(self.instrumental_datalist)
        
        self.song_dict = {
                'supervised': deque([i for i in self.supervised_datalist for _ in range(self.hparams.se)]),
                'semi_supervised': deque([i for i in self.semi_supervised_datalist for _ in range(self.hparams.se)]),
                'instrumental': deque([i for i in self.instrumental_datalist for _ in range(self.hparams.se)]),
                'valid': deque(self.valid_datalist),
                'test': deque(self.test_datalist),
                }
        
    def __dataloader(self, mode):
        """
        self.dataset1: Training dataset
        self.dataset2: semi_supervised dataset
        self.dataset3: Instrumental Dataset
        TODO: If DALI is used, the second dataset should be loaded with multiple songs
        """
        
        if mode == "train":
            # --- Supervised Dataset ---
            supervised_song_name, train_dataset_name = self.song_dict['supervised'].popleft()
            self.song_dict['supervised'].extend([(supervised_song_name, train_dataset_name)])      
            if self.dataset3 != []:
                instrumental_song_name, inst_dataset_name = self.song_dict['instrumental'].popleft()
                self.song_dict['instrumental'].extend([(instrumental_song_name, inst_dataset_name)])
            else:
                instrumental_song_name = None
                inst_dataset_name = None
            supervised_song_dataset = TrainDataset(
                data_path=self.hparams.data_path, dataset1=train_dataset_name, dataset2=inst_dataset_name,
                filename1=supervised_song_name, filename2=instrumental_song_name, mix_ratio=self.hparams.mix_ratio,
                longtail_args={'on_smooth': self.hparams.lt_on_smooth, 'off_smooth': self.hparams.lt_off_smooth},
                device=self.device, use_cp=self.check_cp_available(train_dataset_name) and self.hparams.use_cp, semi=False,
                num_feat=self.hparams.num_feat, k=self.hparams.k
                )
            
            if self.dataset2 != []:
                semi_supervised_song_name, semi_dataset_name = self.song_dict['semi_supervised'].popleft()
                self.song_dict['semi_supervised'].extend([(semi_supervised_song_name, semi_dataset_name)])
                semi_supervised_song_dataset = TrainDataset(
                    data_path=self.hparams.data_path, dataset1=semi_dataset_name, dataset2=None, 
                    filename1=semi_supervised_song_name, filename2=None, mix_ratio=self.hparams.mix_ratio,
                    device=self.device, use_cp=self.hparams.use_cp, semi=True,
                    num_feat=self.hparams.num_feat, k=self.hparams.k
                    )
            
            if self.dataset2 is []:
                self.training_dataset = supervised_song_dataset
            else:
                self.training_dataset = ConcatDataset(supervised_song_dataset, semi_supervised_song_dataset)
            
            return torch.utils.data.DataLoader(self.training_dataset, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle,
                                               num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
            
        elif mode == "valid":
            valid_song_name, valid_dataset_name = self.song_dict['valid'].popleft()
            self.song_dict['valid'].extend([(valid_song_name, valid_dataset_name)])
            self.validation_dataset = EvalDataset(
                data_path=self.hparams.data_path, dataset1=valid_dataset_name, filename1=valid_song_name,
                device=self.device, use_cp=self.check_cp_available(valid_dataset_name) and self.hparams.use_cp, no_pitch=True, use_ground_truth=False,
                num_feat=self.hparams.num_feat, k=self.hparams.k,
                batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory
                )
            return torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                                               num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
    
        elif mode == "test":
            test_song_name, test_dataset_name = self.song_dict['test'].popleft()
            self.song_dict['test'].extend([(test_song_name, test_dataset_name)])
            self.testing_dataset = EvalDataset(
                data_path=self.hparams.data_path, dataset1=test_dataset_name, filename1=test_song_name,
                device=self.device, use_cp=self.check_cp_available(test_dataset_name) and self.hparams.use_cp, no_pitch=False, use_ground_truth=self.hparams.use_ground_truth,
                num_feat=self.hparams.num_feat, k=self.hparams.k,
                batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory
                )
            return torch.utils.data.DataLoader(self.testing_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                                               num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
        
    def train_dataloader(self):
        return self.__dataloader(mode='train')
    
    def val_dataloader(self):
        return self.__dataloader(mode='valid')
    
    def test_dataloader(self):
        return self.__dataloader(mode='test')
    
    def load_from_checkpoint(self, checkpoint_path, use_gpu=True):

        # --- Device ---
        if use_gpu:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            for idx, memory in enumerate(memory_available):
                print(f'cuda:{idx} available memory: {memory}')
            self.device = torch.device(f'cuda:{np.argmax(memory_available)}')
            print(f'Selected cuda:{np.argmax(memory_available)} as device')
            torch.cuda.set_device(int(np.argmax(memory_available)))
        else:
            self.device = torch.device('cpu')

        checkpoint = torch.load(checkpoint_path)
        #self.hparams = argparse.Namespace(**{**checkpoint['hparams'], **self.hparams.__dict__})
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor = amp.initialize(
            self.feature_extractor,
            opt_level=self.hparams.amp_level
        )
        self.feature_extractor.load_state_dict(checkpoint['model'])
        amp.load_state_dict(checkpoint['amp'])

    #def __call__(self, x):
    #    """To use forward like nn.Module"""
    #    return self.forward(x)