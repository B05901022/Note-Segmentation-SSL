# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 05:37:22 2020

@author: Austin Hsu
"""

from src.solver import OnOffsetSolver
from src.trainer import Trainer
import argparse

def train(trainer_hparams, solver_hparams):
    
    trainer = Trainer(trainer_hparams)
    solver = OnOffsetSolver(solver_hparams)
    
    trainer.fit(solver)
    
    trainer.test(solver)
    
    return

def test(trainer_hparams, solver_hparams):
    
    trainer = Trainer(trainer_hparams)
    solver = OnOffsetSolver(solver_hparams)
    
    trainer.test(solver)
    
    return

if __name__ == '__main__':
    
    # --- Solver Args ---
    solver_parser = argparse.ArgumentParser()
    
    solver_parser.add_argument('--model_type', help='Model used', default='PyramidNet_ShakeDrop', type=str)
    solver_parser.add_argument('--loss_type', help='Loss used', default='VAT', type=str)
    solver_parser.add_argument('--dataset1', help='Supervised dataset', default='TONAS', type=str)
    solver_parser.add_argument('--dataset2', help='Semi-supervised dataset', default='Pop_Rhythm', type=str)
    solver_parser.add_argument('--dataset3', help='Instrumental dataset', default='Pop_Rhythm_Instrumental', type=str)
    solver_parser.add_argument('--dataset4', help='Validation dataset', default='None', type=str)
    solver_parser.add_argument('--dataset5', help='Test dataset', default='DALI', type=str)
    solver_parser.add_argument('--mix_ratio', help='Ratio of instrumental dataset to mix with supervised dataset', default=0.5, type=float)
    solver_parser.add_argument('--meta_path', help='Datapath to meta files', default='./meta/', type=str)
    solver_parser.add_argument('--data_path', help='Datapath to wav files', default='../data/', type=str)
    solver_parser.add_argument('--lr', help='Learning rate', default=0.0001, type=float)
    solver_parser.add_argument('--lr_warmup', help='Warmup steps for learning rate', default=40000, type=int)
    solver_parser.add_argument('--max_steps', help='Max training steps', default=240000, type=int)
    solver_parser.add_argument('--se', help='Re-train same sample for how many epochs', default=2, type=int)
    solver_parser.add_argument('--num_feat', help='Numbers of channel dimension', default=9, type=int)
    solver_parser.add_argument('--k', help='Window side from one side (Window size=2*k+1)', default=9, type=int)
    solver_parser.add_argument('--batch_size', help='Batch size', default=64, type=int)
    solver_parser.add_argument('--num_workers', help='Num workers', default=1, type=int)
    solver_parser.add_argument('--shuffle', help='Shuffle dataset or not', action='store_true')
    solver_parser.add_argument('--pin_memory', help='Pin memory', action='store_true')
    solver_parser.add_argument('--use_cp', help='Use CuPy instead of Numpy for data preprocessing', action='store_true')

    solver_parser.set_defaults(
    	shuffle=True,
    	pin_memory=True,
    	use_cp=True,
    )

    solver_hparams = solver_parser.parse_args()

    # --- Trainer Args ---
    trainer_parser = argparse.ArgumentParser()
    trainer_parser.add_argument('--exp_name', help='Name of current experiment', default='Train_0', type=str)
    trainer_parser.add_argument('--log_path', help='Path to save log files', default='./log/', type=str)
    trainer_parser.add_argument('--save_path', help='Path to save weight', default='./checkpoints/', type=str)
    trainer_parser.add_argument('--project', help='Wandb project name', default='note_segmentation', type=str)
    trainer_parser.add_argument('--entity', help='Wandb user name', default='austinhsu', type=str)
    trainer_parser.add_argument('--amp_level', help='Amp level for mixed precision training', default='O0', type=str)
    trainer_parser.add_argument('--accumulate_grad_batches', help='Gradient accumulation', default=1, type=int)
    trainer_parser.add_argument('--use_amp', help='Use and import amp for mixed precision training', action='store_true')
    trainer_parser.add_argument('--train', help='Run in train mode', action='store_true')
    trainer_parser.add_argument('--test', help='Run in test mode', action='store_true')

    trainer_parser.set_defaults(
        use_amp=False,
    	train=True,
    	test=False
    )

    trainer_hparams = trainer_parser.parse_args()

    if trainer_hparams.train:
        train(trainer_hparams, solver_hparams)

    if trainer_hparams.test:
        test(trainer_hparams, solver_hparams)    