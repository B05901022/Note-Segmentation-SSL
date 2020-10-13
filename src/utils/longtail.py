# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:21:35 2020

@author: Austin Hsu
"""

import numpy as np

def unsmooth(onoffset: np.ndarray, second_order: bool = True) -> np.ndarray:
    smooth_filter = np.array([0,1,1,1,1,1,0])
    unsmooth_onoffset = np.convolve(onoffset, smooth_filter)[3:-3]
    unsmooth_onoffset = (unsmooth_onoffset==5).astype(onoffset.dtype)
    
    if second_order:
        # only pick the leftmost and rightmost of continue onset/offset
        # ex: 0,0,1,1,1,0,0 ==> 0,0,1,0,1,0,0
        sec_filter = np.array([1,1,1])
        sec_unsmooth_onoffset = np.convolve(unsmooth_onoffset, sec_filter)[1:-1]
        sec_unsmooth_onoffset = (sec_unsmooth_onoffset<3).astype(onoffset.dtype)
        unsmooth_onoffset = unsmooth_onoffset*sec_unsmooth_onoffset
    
    return unsmooth_onoffset

def fix_pair_mismatch(onset_pos: np.ndarray, offset_pos: np.ndarray) -> (np.ndarray, np.ndarray):
    fix_onset = []
    fix_offset = []
    pos_o = 0
    pos_f = 0
    while pos_o < onset_pos.shape[0] and pos_f < offset_pos.shape[0]:
        if onset_pos[pos_o] <= offset_pos[pos_f]:
            if pos_o < onset_pos.shape[0] -1:
                if onset_pos[pos_o+1] < offset_pos[pos_f]:
                    # miss one offset
                    fix_onset.append(onset_pos[pos_o])
                    fix_offset.append(onset_pos[pos_o+1])
                    pos_o += 1
                else:
                    # correct case
                    fix_onset.append(onset_pos[pos_o])
                    fix_offset.append(offset_pos[pos_f])
                    pos_o += 1
                    pos_f += 1
            else:
                fix_onset.append(onset_pos[pos_o])
                fix_offset.append(offset_pos[pos_f])
                pos_o += 1
                pos_f += 1
        else:
            if pos_f > 0:
                # miss one onset
                fix_onset.append(offset_pos[pos_f-1])
                fix_offset.append(offset_pos[pos_f])
                pos_f += 1
            else:
                # discard first offset if it is earlier than first onset
                pos_f += 1
    assert(len(fix_onset) == len(fix_offset))
    return np.array(fix_onset), np.array(fix_offset)

def longtail(sdt: np.ndarray, on_smooth: bool = False, off_smooth: bool = True) -> np.ndarray:
    """
    sdt: (time_length, 6) (Silence, Duration, Onset_neg, Onset, Offset_neg, Offset)
    """
    
    unsmooth_onset = unsmooth(sdt[:,3])
    unsmooth_offset = unsmooth(sdt[:,5])
    
    onset_pos = np.where(unsmooth_onset)[0]
    offset_pos = np.where(unsmooth_offset)[0]
    
    onset_pos, offset_pos = fix_pair_mismatch(onset_pos, offset_pos)
    unsmooth_onset = np.zeros(unsmooth_onset.shape)
    unsmooth_offset = np.zeros(unsmooth_offset.shape)
    unsmooth_onset[onset_pos] = 1.
    unsmooth_offset[offset_pos] = 1.
    
    #print(f'{(onset_pos<=offset_pos).sum()} / {onset_pos.shape[0]}')
    #print(np.where(onset_pos>offset_pos)[0])
    assert((onset_pos<=offset_pos).sum() == onset_pos.shape[0])
    
    modified_duration = sdt[:,1].astype(np.float)
    modified_onset = unsmooth_onset.astype(np.float)
    modified_offset = unsmooth_offset.astype(np.float)
        
    if on_smooth:
        on_longtail = list(zip(np.hstack((np.array([0]), offset_pos[:-1])), onset_pos))
        for offset, onset in on_longtail:
            modified_onset[offset:onset] = np.linspace(0,1,num=onset-offset)
            modified_duration[offset:onset] = np.linspace(0,1,num=onset-offset)
    
    if off_smooth:
        off_longtail = list(zip(offset_pos, np.hstack((onset_pos[1:], np.array(sdt.shape[0])))))
        for offset, onset in off_longtail:
            modified_offset[offset:onset] = np.linspace(1,0,num=onset-offset)
            modified_duration[offset:onset] = np.linspace(1,0,num=onset-offset)
        
    # smooth
    #for onset in onset_pos:
    #    modified_onset[max(0, onset-2): min(sdt.shape[0], onset+4)] = 1
    #for offset in offset_pos:
    #    modified_offset[max(0, offset-2): min(sdt.shape[0], offset+4)] = 1
    modified_onset = np.clip(modified_onset + sdt[:,3], 0, 1)
    modified_offset = np.clip(modified_offset + sdt[:,5], 0, 1)
    
    modified_sdt = np.zeros(sdt.shape)
    modified_sdt[:,1] = modified_duration
    modified_sdt[:,3] = modified_onset
    modified_sdt[:,5] = modified_offset
    modified_sdt[:,0] = (1-modified_sdt[:,1])
    modified_sdt[:,2] = (1-modified_sdt[:,3])
    modified_sdt[:,4] = (1-modified_sdt[:,5])
    
    return modified_sdt