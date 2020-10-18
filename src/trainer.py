# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 00:52:22 2020

@author: Austin Hsu
"""

from src.utils.tools import AttributeDict
from src.utils.profiler import SimpleProfiler
import wandb
import torch
import random
import os
import numpy as np
import apex.amp as amp
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict

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
        
        # --- Random Seed ---
        self.seed = 17
        self._setup_seed(self.seed)
        
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

        # --- Profiler ---
        self.profiler = SimpleProfiler()

        self.best_epoch = None # To check if test right after train or test individually
    
    def _setup_seed(self, seed):
        """Set random seed."""
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True

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
            torch.cuda.set_device(int(np.argmax(memory_available)))
        else:
            self.device = torch.device('cpu')
        solver.device = self.device

        # --- Init ---
        # self.logger.watch(solver.feature_extractor)
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

        # --- Graceful Ctrl + C ---
        try:
            while self.train_step < solver.hparams.max_steps and self.train_epoch < solver.hparams.max_epoch:
                
                print(f"Epoch: {self.train_epoch:2d}")
                
                # --- Init Song List (Shuffle) ---
                solver.initsonglist()

                # --- Train Loop ---
                solver.feature_extractor.train()
                self.song_number = len(solver.supervised_datalist)
                avg_train_loss = 0
                for train_update in self.profiler.profile_iterable(range(self.song_number*solver.hparams.se), 'Train Loop (per song)'):
                    avg_train_loss += self.per_song_train_loop(solver, optimizer, scheduler, train_update, amp)
                avg_train_loss /= self.song_number*solver.hparams.se
                self.logger.log({'avg_train_loss': avg_train_loss})
                self.logger.log({'Epoch': self.train_epoch})

                if not self.hparams.skip_val:
                    # --- Valid Loop ---
                    solver.feature_extractor.eval()
                    self.song_number = len(solver.valid_datalist)
                    avg_valid_loss = 0
                    for valid_update in self.profiler.profile_iterable(range(self.song_number), 'Validation Loop (per song)'):
                        avg_valid_loss += self.per_song_valid_loop(solver, valid_update)
                    avg_valid_loss /= self.song_number
                    self.logger.log({'avg_valid_loss': avg_valid_loss})
                    
                    # --- Checkpoint ---
                    if avg_valid_loss < min_loss:
                        with self.profiler.profile('Save Model'):
                            print('Renewing best model ...')
                            min_loss = avg_valid_loss
                            solver.best_epoch = self.train_epoch
                            check_point = {
                                'model': solver.feature_extractor.state_dict(),
                                'amp': amp.state_dict(),
                                'hparams': solver.hparams,
                                'best_epoch': self.train_epoch,
                                'best_loss': avg_valid_loss
                                }
                            torch.save(check_point, os.path.join(self.hparams.save_path, f'epoch={self.train_epoch}.pt'))
                            if self.best_epoch is not None:
                                os.replace(
                                    os.path.join(self.hparams.save_path, f'epoch={self.best_epoch}.pt'),
                                    os.path.join(self.hparams.save_path, f'epoch={self.train_epoch}.pt')
                                    )
                            self.best_epoch = self.train_epoch
                else:
                    # --- Checkpoint ---
                    if avg_train_loss < min_loss:
                        with self.profiler.profile('Save Model'):
                            print('Renewing best model ...')
                            min_loss = avg_train_loss
                            solver.best_epoch = self.train_epoch
                            check_point = {
                                'model': solver.feature_extractor.state_dict(),
                                'amp': amp.state_dict(),
                                'hparams': solver.hparams,
                                'best_epoch': self.train_epoch,
                                'best_loss': avg_train_loss
                                }
                            torch.save(check_point, os.path.join(self.hparams.save_path, f'epoch={self.train_epoch}.pt'))
                            if self.best_epoch is not None:
                                os.replace(
                                    os.path.join(self.hparams.save_path, f'epoch={self.best_epoch}.pt'),
                                    os.path.join(self.hparams.save_path, f'epoch={self.train_epoch}.pt')
                                    )
                            self.best_epoch = self.train_epoch
                self.train_epoch += 1

        except KeyboardInterrupt:
            self.profiler.describe()
            if self.best_epoch is None:
                print(f'No model saved.')
            else:
                print(f'Best model: {self.best_epoch}.pt')
            print('Exiting from training early.')

        # --- Profiler Summarization ---
        self.profiler.describe()
        
    def test(self, solver):
        """Does testing loop."""
        # --- Init ---
        solver.initsonglist()
        if self.best_epoch is not None:
            solver.load_from_checkpoint(os.path.join(self.hparams.save_path, f'epoch={self.best_epoch}.pt'), use_gpu=self.hparams.use_gpu)
        else:
            self.logger.watch(solver.feature_extractor)
        self.device = solver.device
        solver.test_no_offset = self.hparams.test_no_offset

        # --- Log Dict ---
        log_dict = OrderedDict({
            'Onset_F1': [],
            'Offset_F1': [],
            'Transcription_F1': [],
            'Transcription_F1(No_Offset)': [],
            'Transcription_F1(Onset_100ms)': [],
            'Transcription_F1(No_Offset, Onset_100ms)': [],
            'Onset_Precision': [],
            'Offset_Precision': [],
            'Transcription_Precision': [],
            'Transcription_Precision(No_Offset)': [],
            'Transcription_Precision(Onset_100ms)': [],
            'Transcription_Precision(No_Offset, Onset_100ms)': [],
            'Onset_Recall': [],
            'Offset_Recall': [],
            'Transcription_Recall': [],
            'Transcription_Recall(No_Offset)': [],
            'Transcription_Recall(Onset_100ms)': [],
            'Transcription_Precision(No_Offset, Onset_100ms)': [],
            'Average_Overlap_Ratio': [],
            'Average_Overlap_Ratio(No_Offset)': [],
            'Average_Overlap_Ratio(Onset_100ms)': [],
            'Average_Overlap_Ratio(No_Offset, Onset_100ms)': [],
            'Conflict_Ratio': [],
            'Note_Accuracy': [],
        })
        
        solver.feature_extractor.eval()
        self.song_number = len(solver.test_datalist)
        for test_update in self.profiler.profile_iterable(range(self.song_number), 'Test Loop (per song)'):
            test_outputs = self.per_song_test_loop(solver, test_update)
            for test_key in log_dict.keys():
                log_dict[test_key].append(test_outputs.log[test_key])
        

        for test_key in log_dict.keys():
            log_dict[test_key] = np.mean(log_dict[test_key])
        self.logger.log(log_dict)

        # --- Profiler Summarization ---
        self.profiler.describe()
        
    def per_song_train_loop(self, solver, optimizer, scheduler, update, amp):
        with self.profiler.profile('Get Train DataLoader'):
            train_dataloader = solver.train_dataloader()
        tqdm_iterator = tqdm(total=len(train_dataloader), position=0, leave=True)
        train_loss = []
        for batch_idx, batch in self.profiler.profile_iterable(enumerate(train_dataloader), 'Train Loop (per segment)'):

            # --- To Device ---
            if solver.dataset2 is not None:
                batch[0] = [batch[0][0].to(self.device), batch[0][1].to(self.device, non_blocking=True)] # Train dataset
                batch[1] = batch[1].to(self.device) # Semi dataset
            else:
                batch = [batch[0].to(self.device), batch[1].to(self.device, non_blocking=True)] # Train dataset

            # --- Train Step ---
            with self.profiler.profile('Train Forward'):
                get_train_output = solver.training_step(batch, batch_idx)
            get_train_output = AttributeDict(get_train_output)
            
            # --- Backward ---
            with self.profiler.profile('Train Backward'):
                with amp.scale_loss(get_train_output.loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            # --- Update Model ---
            if (self.train_step+1)%self.hparams.accumulate_grad_batches == 0:
                with self.profiler.profile('Optimizer Step'):
                    optimizer.step()
                optimizer.zero_grad()
            if scheduler is not None:
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
            
            # --- Calculate Mean Train Loss ---
            train_loss.append(get_train_output.log['train_loss'])

            self.train_step += 1
            
        tqdm_iterator.close()
        
        return np.mean(train_loss)
        
    def per_song_valid_loop(self, solver, update):
        with self.profiler.profile('Get Validation DataLoader'):
            valid_dataloader = solver.valid_dataloader()
        tqdm_iterator = tqdm(total=len(valid_dataloader), position=0, leave=True)
        outputs = []
        for batch_idx, batch in self.profiler.profile_iterable(enumerate(valid_dataloader), 'Validation Loop (per segment)'):
            
            # --- To Device ---
            batch = [i.to(self.device) for i in batch]

            # --- Valid Step ---
            with self.profiler.profile('Validation Forward'):
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
        with self.profiler.profile('Validation End'):
            result = solver.validation_epoch_end(outputs)
        result = AttributeDict(result)
        self.logger.log(result.progress_bar)
        
        return result.progress_bar['val_loss']
    
    def per_song_test_loop(self, solver, update):
        with self.profiler.profile('Get Test DataLoader'):
            test_dataloader = solver.test_dataloader()
        tqdm_iterator = tqdm(
            desc=f"Song {update:3d}/{self.song_number:3d}",
            total=len(test_dataloader),
            position=0,
            leave=True
        )
        outputs = []
        for batch_idx, batch in self.profiler.profile_iterable(enumerate(test_dataloader), 'Test Loop (per segment)'):
            
            # --- To Device ---
            batch = [i.to(self.device) for i in batch]

            # --- Test Step ---
            with self.profiler.profile('Test Forward'):
                get_test_output = solver.test_step(batch, batch_idx)
            get_test_output = AttributeDict(get_test_output)
            
            # --- Progress Bar ---
            tqdm_iterator.update()
            
            # --- Logger ---
            outputs.append(get_test_output.sdt)
                        
        tqdm_iterator.close()
        
        # --- Test End ---
        outputs = torch.cat(outputs)
        with self.profiler.profile('Test End'):
            result = solver.test_epoch_end(outputs)
        result = AttributeDict(result)
        for log_key in result.log.keys():
            self.logger.log({f'{log_key}_sep': result.log[log_key]})
        #self.logger.log(result.log)
        
        return result