# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:38:00 2020

@author: Austin Hsu
"""

from src.solver import OnOffsetSolver
from src.dataset import *

import pytorch_lightning as pl
import os
import argparse
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger

def train(hparams):
    
    # --- Logger ---
    wandb_logger = WandbLogger(
            name=hparams.exp_name,
            save_dir=hparams.log_path,
            project='note_segmentation',
            entity='austinhsu',
            )
    
    # --- LR Logger ---
    lr_logger = LearningRateLogger()
    callbacks = [lr_logger]
    
    # --- Checkpoint ---
    checkpoint_callback = ModelCheckpoint(
            filepath=hparams.save_path,
            save_top_k=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix='',
            )
    
    # --- Model ---
    if hparams.conti:
        model = OnOffsetSolver.load_from_checkpoint(os.path.join(hparams.save_path, hparams.checkpoint_name))
    else:
        model = OnOffsetSolver(hparams)
    
    # --- Trainer ---
    trainer = pl.Trainer(
            accumulate_grad_batches=hparams.accumulate_grad_batches,
            max_epochs=hparams.epochs,
            gpus=hparams.gpus,
            auto_select_gpus=hparams.auto_select_gpus,
            distributed_backend=hparams.distributed_backend if hparams.distributed_backend != "None" else None,
            amp_level='O1' if hparams.use_amp else 'O0',
            precision=16 if hparams.use_16bit or hparams.use_amp else 32,
            benchmark=hparams.benchmark,
            reload_dataloaders_every_epoch=True, # Ensure to train different songs
            default_save_path=hparams.save_path,
            logger=wandb_logger,
            val_check_interval=hparams.val_check_interval,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
            
            fast_dev_run=False,
            max_steps=hparams.max_steps,
            #max_steps=20, # For fast training test
            weights_summary='top',
            train_pecent_check=1.0,
            val_percent_check=1.0,
            test_percent_check=1.0,
            log_gpu_memory='min_max',
            track_grad_norm=-1,
            profiler=True,
            #auto_scale_batch_size='binsearch',
            )
    trainer.fit(model)
    
    # --- Test ---
    trainer.test()
    
    return

def test(hparams):
    
    # --- Logger ---
    wandb_logger = WandbLogger(
            name=hparams.exp_name,
            save_dir=hparams.log_path,
            project='note_segmentation',
            entity='austinhsu',
            )
    
    # --- DataLoader ---
    dataset = ...
    TestLoader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=hparams.batch_size, 
            shuffle=False,
            num_workers=hparams.num_workers, 
            pin_memory=hparams.pin_memory)
    
    # --- Checkpoint ---
    model = OnOffsetSolver.load_from_checkpoint(os.path.join(hparams.save_path, hparams.checkpoint_name))
    
    # --- Test ---
    trainer = pl.Trainer(
            gpus=hparams.gpus,
            auto_select_gpus=hparams.auto_select_gpus,
            distributed_backend=hparams.distributed_backend if hparams.distributed_backend != "None" else None,
            amp_level='O1' if hparams.use_amp else 'O0',
            precision=16 if hparams.use_16bit or hparams.use_amp else 32,
            benchmark=hparams.benchmark,
            log_gpu_memory='min_max',
            logger=wandb_logger,
            )
    trainer.test(model, test_dataloaders=TestLoader)
    
    return

if __name__ == '__main__':
    
    # --- Args ---
    parent_parser = argparse.ArgumentParser()
    
    parent_parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parent_parser.add_argument('--gpus', default=1, type=int)
    parent_parser.add_argument('--distributed_backend', default='None', type=str)
    parent_parser.add_argument('--auto_select_gpus', action='store_true')
    parent_parser.add_argument('--use_16bit', action='store_true')
    parent_parser.add_argument('--use_amp', action='store_true')
    parent_parser.add_argument('--benchmark', action='store_true')
    
    parent_parser.add_argument('--save_path', default='./checkpoints/', type=str)
    parent_parser.add_argument('--log_path', default='./log/', type=str)
    parent_parser.add_argument('--exp_name', default='default', type=str)
    parent_parser.add_argument('--val_check_interval', default=0.25, type=float)
    
    parent_parser.add_argument('--train', action='store_true')
    parent_parser.add_argument('--conti', action='store_true')
    parent_parser.add_argument('--test', action='store_true')
    parent_parser.add_argument('--checkpoint_name', default='epoch=10.ckpt', type=str)
    
    parent_parser.set_defaults(auto_select_gpus=True,
                               use_16bit=False,
                               use_amp=False,
                               benchmark=True,
                               )
    
    parser = OnOffsetSolver.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()
    
    # --- Train ---
    if hyperparams.train:
        train(hyperparams)
    
    # --- Test ---
    if hyperparams.test:
        test(hyperparams)
    