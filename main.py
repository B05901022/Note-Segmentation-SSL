# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 05:37:22 2020

@author: Austin Hsu
"""

from src.solver import OnOffsetSolver
from src.trainer import Trainer
import argparse
import os

def train(hparams):
    
    trainer = Trainer(hparams)
    solver = OnOffsetSolver(hparams)
    
    trainer.fit(solver)
    
    trainer.test(solver)
    
    return

def test(hparams):
    
    trainer = Trainer(hparams)
    solver = OnOffsetSolver(hparams)

    solver.load_from_checkpoint(os.path.join(hparams.save_path, hparams.checkpoint_name), use_gpu=hparams.use_gpu)
    
    trainer.test(solver)
    
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # --- Solver Args ---
    parser.add_argument('--model_type', help='Model used', default='PyramidNet_ShakeDrop', type=str)
    parser.add_argument('--loss_type', help='Loss used', default='VAT', type=str)
    parser.add_argument('--dataset1', help='Supervised dataset', default='TONAS', type=str)
    parser.add_argument('--dataset2', help='Semi-supervised dataset', default='Pop_Rhythm', type=str)
    parser.add_argument('--dataset3', help='Instrumental dataset', default='Pop_Rhythm_Instrumental', type=str)
    parser.add_argument('--dataset4', help='Validation dataset', default='None', type=str)
    parser.add_argument('--dataset5', help='Test dataset', default='DALI', type=str)
    parser.add_argument('--mix_ratio', help='Ratio of instrumental dataset to mix with supervised dataset', default=0.5, type=float)
    parser.add_argument('--meta_path', help='Datapath to meta files', default='./meta/', type=str)
    parser.add_argument('--data_path', help='Datapath to wav files', default='../data/', type=str)
    parser.add_argument('--lr', help='Learning rate', default=0.0001, type=float)
    parser.add_argument('--lr_warmup', help='Warmup steps for learning rate', default=40000, type=int)
    parser.add_argument('--max_steps', help='Max training steps', default=240000, type=int)
    parser.add_argument('--max_epoch', help='Max training epoch', default=20, type=int)
    parser.add_argument('--se', help='Re-train same sample for how many epochs', default=2, type=int)
    parser.add_argument('--num_feat', help='Numbers of channel dimension', default=9, type=int)
    parser.add_argument('--k', help='Window side from one side (Window size=2*k+1)', default=9, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=64, type=int)
    parser.add_argument('--num_workers', help='Num workers', default=1, type=int)
    parser.add_argument('--use_ground_truth', help='Use ground truth pitch file, otherwise use patch CNN for pitch extraction', action='store_true')
    parser.add_argument('--shuffle', help='Shuffle dataset or not', action='store_true')
    parser.add_argument('--pin_memory', help='Pin memory', action='store_true')
    parser.add_argument('--use_cp', help='Use CuPy instead of Numpy for data preprocessing', action='store_true')

    # --- Trainer Args ---
    parser.add_argument('--exp_name', help='Name of current experiment', default='Train_0', type=str)
    parser.add_argument('--log_path', help='Path to save log files', default='./log/', type=str)
    parser.add_argument('--save_path', help='Path to save weight', default='./checkpoints/', type=str)
    parser.add_argument('--project', help='Wandb project name', default='note_segmentation', type=str)
    parser.add_argument('--entity', help='Wandb user name', default='austinhsu', type=str)
    parser.add_argument('--checkpoint_name', help='Checkpoint .pt file name', default='epoch=0', type=str)
    parser.add_argument('--amp_level', help='Amp level for mixed precision training', default='O0', type=str)
    parser.add_argument('--accumulate_grad_batches', help='Gradient accumulation', default=1, type=int)
    parser.add_argument('--skip_val', help='Skip validation loop, use train loss for best model', action='store_true')
    parser.add_argument('--use_gpu', help='Use gpu for training', action='store_true')
    parser.add_argument('--use_amp', help='Use automatic mixed precision (amp) for training', action='store_true')
    parser.add_argument('--test_no_offset', help='Test without offset_ratio, recommended for offset-unstable datasets (like DALI)', action='store_true')
    parser.add_argument('--train', help='Run in train mode', action='store_true')
    parser.add_argument('--test', help='Run in test mode', action='store_true')

    #parser.set_defaults(
    #    # --- Solver Args ---
    #    use_ground_truth=False,
    # 	 shuffle=True,
    # 	 pin_memory=True,
    # 	 use_cp=True,

    #    # --- Trainer Args ---
    #    skip_val=False,
    #    use_gpu=True,
    #    train=True,
    #    test=False,
    #)

    hyperparams = parser.parse_args()

    if hyperparams.train:
        train(hyperparams)

    if hyperparams.test:
        test(hyperparams)    