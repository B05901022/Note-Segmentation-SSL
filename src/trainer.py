# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 00:52:22 2020

@author: Austin Hsu
"""

from src.utils.tools import AttributeDict
import wandb
import torch
import os
import numpy as np
import apex.amp as amp
from tqdm import tqdm

class Trainer:
    """
    Args:
        exp_name: Name of current experiment
        log_path: Path to save log files
        save_path: Path to save weight
        project: Wandb project name
        entity: Wandb user name
        amp_level: Amp level for mix precision training
        accumulate_grad_batches: Gradient accumulation
    """
    def __init__(self, hparams):
        self.hparams = hparams
        
        # --- Logger ---
        self.logger = wandb.init(
            name=self.hparams.exp_name,
            dir=self.hparams.log_path,
            project=self.hparams.project,
            entity=self.hparams.entity,
            anonymous=None,
            reinit=True,
            id=None,
            resume='allow',
            tags=None,
            group=None
            )
        
    def fit(self, solver):
        """Does training training loop."""
        
        # --- Device ---
        if self.hparams.use_gpu:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            for idx, memory in enumerate(memory_available):
                print(f'cuda:{idx} available memory: {memory}')
            self.device = torch.device(f'cuda:{np.argmax(memory_available)}')
            print(f'Selected cuda:{np.argmax(memory_available)} as device')
        else:
            self.device = torch.device('cpu')
        solver.device = self.device

        # --- Init ---
        self.logger.watch(solver.feature_extractor)
        solver.initsonglist()
        solver.feature_extractor = solver.feature_extractor.to(self.device)
        optimizer, scheduler = solver.configure_optimizers()
        solver.feature_extractor, optimizer = amp.initialize(
            solver.feature_extractor,
            optimizer,
            opt_level=self.hparams.amp_level
        )
        optimizer.zero_grad()
        min_loss = 10000.0
        
        self.train_step = 0
        self.train_epoch = 0
        while self.train_step < solver.hparams.max_steps:
            
            print(f"Epoch: {self.train_epoch:2d}")
            
            # --- Train Loop ---
            solver.feature_extractor.train()
            self.song_number = len(solver.supervised_datalist)
            for train_update in range(self.song_number*solver.hparams.se):
                _ = self.per_song_train_loop(solver, optimizer, scheduler, train_update, amp)
                
            # --- Valid Loop ---
            solver.feature_extractor.eval()
            self.song_number = len(solver.valid_datalist)
            avg_valid_loss = 0
            for valid_update in range(self.song_number):
                avg_valid_loss += self.per_song_valid_loop(solver, valid_update)
            avg_valid_loss /= self.song_number
            self.logger.log({'avg_valid_loss': avg_valid_loss})
            
            # --- Checkpoint ---
            if avg_valid_loss < min_loss:
                print('Renewing best model ...')
                min_loss = avg_valid_loss
                check_point = {
                    'model': solver.feature_extractor.state_dict(),
                    'amp': amp.state_dict(),
                    'hparams': solver.hparams
                    }
                torch.save(check_point, self.hparams.save_path+'.pt')
            
            self.train_epoch += 1
        
        # --- Test Loop ---
        self.test(solver)
        
    def test(self, solver):
        """Does testing loop."""
        # --- Init ---
        self.logger.watch(solver.feature_extractor)
        solver.initsonglist()
        self.device = solver.device
        
        # --- Log Dict ---
        log_dict = {
            'Onset_F1': [],
            'Offset_F1': [],
            'Transcription_F1': [],
            'Onset_Precision': [],
            'Offset_Precision': [],
            'Transcription_Precision': [],
            'Onset_Recall': [],
            'Offset_Recall': [],
            'Transcription_Recall': [],
            'Average_Overlap_Ratio': [],
            'Conflict_Ratio': [],
        }
        
        solver.feature_extractor.eval()
        self.song_number = len(solver.test_datalist)
        for test_update in range(self.song_number):
            test_outputs = self.per_song_test_loop(solver, test_update)
            for test_key in log_dict.keys():
                log_dict[test_key].append(test_outputs[test_key])
        
        for test_key in log_dict.keys():
            log_dict[test_key] = np.mean(log_dict[test_key])
        self.logger.log(log_dict)
        
    def per_song_train_loop(self, solver, optimizer, scheduler, update, amp):
        train_dataloader = solver.train_dataloader()
        tqdm_iterator = tqdm(total=len(train_dataloader))
        for batch_idx, batch in enumerate(train_dataloader):

            if solver.dataset2 is not None:
                batch[0] = [i.to(self.device) for i in batch[0]] # Train dataset
                batch[1] = batch[1].to(self.device) # Semi dataset
            else:
                batch = [i.to(self.device) for i in batch] # Train dataset

            # --- Train Step ---
            get_train_output = solver.training_step(batch, batch_idx)
            get_train_output = AttributeDict(get_train_output)
            
            # --- Update Model ---
            with amp.scale_loss(get_train_output.loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (self.train_step+1)%self.hparams.accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            
            # --- Progress Bar ---
            tqdm_iterator.set_description_str(
                f"Song {update//solver.hparams.se:3d}/{self.song_number:3d} "
                f"SE {update%solver.hparams.se:1d}"
                )
            tqdm_iterator.set_postfix_str(str(get_train_output.progress_bar))
            tqdm_iterator.update()
            
            # --- Logger ---
            self.logger.log(get_train_output.log)
            self.logger.log({'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            
            self.train_step += 1
            
        tqdm_iterator.close()
        
        return None
        
    def per_song_valid_loop(self, solver, update):
        valid_dataloader = solver.valid_dataloader()
        tqdm_iterator = tqdm(total=len(valid_dataloader))
        outputs = []
        for batch_idx, batch in enumerate(valid_dataloader):
            
            batch = [i.to(self.device) for i in batch]

            # --- Valid Step ---
            get_valid_output = solver.validation_step(batch, batch_idx)
            get_valid_output = AttributeDict(get_valid_output)
            
            # --- Progress Bar ---
            tqdm_iterator.set_description_str(f"Song {update:3d}/{self.song_number:3d}")
            tqdm_iterator.set_postfix_str(str(get_valid_output.progress_bar))
            tqdm_iterator.update()
            
            # --- Logger ---
            self.logger.log(get_valid_output.log)
            outputs.append(get_valid_output.progress_bar)
                        
        tqdm_iterator.close()
        
        # --- Valid End ---
        result = solver.validation_epoch_end(outputs)
        result = AttributeDict(result)
        self.logger.log(result.progress_bar)
        
        return result.progress_bar['val_loss']
    
    def per_song_test_loop(self, solver, update):
        test_dataloader = solver.test_dataloader()
        tqdm_iterator = tqdm(
            desc=f"Song {update:3d}/{self.song_number:3d}",
            total=len(test_dataloader)
        )
        outputs = []
        for batch_idx, batch in enumerate(test_dataloader):
            
            batch = [i.to(self.device) for i in batch]

            # --- Test Step ---
            get_test_output = solver.test_step(batch, batch_idx)
            get_test_output = AttributeDict(get_test_output)
            
            # --- Progress Bar ---
            tqdm_iterator.update()
            
            # --- Logger ---
            outputs.append(get_test_output.sdt)
                        
        tqdm_iterator.close()
        
        # --- Test End ---
        outputs = torch.stack(outputs)
        result = solver.test_epoch_end(outputs)
        result = AttributeDict(result)
        self.logger.log(result.log)
        
        return result