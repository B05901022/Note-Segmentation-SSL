# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 01:57:14 2020

@author: Austin Hsu
"""

import soundfile as sf
import numpy as np
import scipy.fftpack
import scipy.io.wavfile
import scipy.signal
import argparse
import os
import torch
from src.model.Patch_CNN import KitModel

def read_file(filename: str) -> (np.array, int):
    """Read files"""
    # --- Load File ---
    sample_rate, audio = scipy.io.wavfile.read(filename)
    audio = audio.astype(np.float32)
    if len(audio.shape)==2:
        audio = audio.mean(axis=-1)
        
    #  --- Resample ---
    if sample_rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, sample_rate)
    return audio

def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/float(Hop))*Hop, Hop)
    N = int(fs/float(fr))
    window_size = len(h)
    f = fs*np.linspace(0, 0.5, round(N/2), endpoint=True)
    Lh = int(np.floor(float(window_size-1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)     
        
    for icol in range(0, len(t)):
        ti = int(t[icol])           
        tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                        int(min([round(N/2.0)-1, Lh, len(x)-ti])))
        indices = np.mod(N + tau, N) + 1                                             
        tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                /np.linalg.norm(h[Lh+tau-1])           
                            
    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))  
    return tfr, f, t, N

def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g!=0:
        X[X<0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X

def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        l = int(round(central_freq[i-1]/fr))
        r = int(round(central_freq[i+1]/fr)+1)
        #rounding1
        if l >= r-1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq

def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1/(q+1e-8)
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
            if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

    tfrL = np.dot(freq_band_transformation, ceps)
    return tfrL, central_freq

def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    tfr, f, t, N = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr # original STFT
    ceps = np.zeros(tfr.shape)

    for gc in range(1, 3):
        if np.remainder(gc, 2) == 1:
            tc_idx = round(fs*tc)
            ceps = np.real(np.fft.fft(tfr, axis=0))/np.sqrt(N)
            ceps = nonlinear_func(ceps, g[gc], tc_idx)
        else:
            fc_idx = round(fc/fr)
            tfr = np.real(np.fft.fft(ceps, axis=0))/np.sqrt(N)
            tfr = nonlinear_func(tfr, g[gc], fc_idx)

    tfr0 = tfr0[:int(round(N/2)),:]
    tfr = tfr[:int(round(N/2)),:]
    ceps = ceps[:int(round(N/2)),:]

    HighFreqIdx = int(round((1/tc)/fr)+1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx,:]
    tfr = tfr[:HighFreqIdx,:]
    HighQuefIdx = int(round(fs/fc)+1)
    q = np.arange(HighQuefIdx)/float(fs)
    ceps = ceps[:HighQuefIdx,:]
    
    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies 

def melody_feature_extraction(x):
    # --- Args ---
    fs = 16000.0 # sampling frequency
    x = x.astype('float32')
    Hop = 320 # hop size (in sample)
    h = scipy.signal.blackmanharris(2049) # window size
    fr = 2.0 # frequency resolution
    fc = 80.0 # the frequency of the lowest pitch
    tc = 1/1000.0 # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    NumPerOctave = 48 # Number of bins per octave
    
    tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
    Z = tfrLF * tfrLQ
    return Z, t, CenFreq

def patch_extraction(Z, patch_size, th):
    # Z is the input spectrogram or any kind of time-frequency representation
    M, N = np.shape(Z)    
    half_ps = int(np.floor(float(patch_size)/2)) #12

    Z = np.pad(Z, ((0, half_ps), (half_ps, half_ps)))
    
    M, N = np.shape(Z)
    
    data = []
    mapping = []
    counter = 0
    for t_idx in range(half_ps, N-half_ps):
        LOCS = findpeaks(Z[:,t_idx], th)
        for mm in range(0, len(LOCS)):
            if LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter<300000:# and PKS[mm]> 0.5*max(Z[:,t_idx]):
                patch = Z[np.ix_(range(LOCS[mm]-half_ps, LOCS[mm]+half_ps+1), range(t_idx-half_ps, t_idx+half_ps+1))]
                data.append(patch)
                mapping.append((LOCS[mm], t_idx))
                counter = counter + 1
            elif LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter>=300000:
                print('Out of the biggest size. Please shorten the input audio.')
                
    data = np.array(data[:-1])
    mapping = np.array(mapping[:-1])
    Z = Z[:M-half_ps,:]
    return data, mapping, half_ps, N, Z

class temp_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).unsqueeze(1).float()
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]

def patch_prediction(modelname, data, patch_size,
                     batch_size, num_workers, pin_memory):
    
    modelname = os.path.join('./',modelname+'.npy')
    model = KitModel(weight_file=modelname).cuda()
    dataset = temp_dataset(data=data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    pred = []
    model = model.eval()
    for b_id, b_x in enumerate(dataloader):
        b_x = b_x.cuda()
        pred.append(model(b_x).detach().cpu().numpy())
    return np.concatenate(pred)

def contour_prediction(mapping, pred, N, half_ps, Z, t, CenFreq, max_method):
    PredContour = np.zeros(N)

    pred = pred[:,1]
    pred_idx = np.where(pred>0.5)
    MM = mapping[pred_idx[0],:]
    pred_prob = pred[pred_idx[0]]
    MM = np.append(MM, np.reshape(pred_prob, [len(pred_prob),1]), axis=1)
    MM = MM[MM[:,1].argsort()]    
    
    for t_idx in range(half_ps, N-half_ps):
        Candidate = MM[np.where(MM[:,1]==t_idx)[0],:]
        if Candidate.shape[0] >= 2:
            if max_method == 'posterior':
                fi = np.where(Candidate[:,2]==np.max(Candidate[:,2]))
                fi = fi[0]
            elif max_method == 'prior':
                fi = Z[Candidate[:,0].astype('int'),t_idx].argmax(axis=0)
            fi = fi.astype('int')
            PredContour[Candidate[fi,1].astype('int')] = Candidate[fi,0] 
        elif Candidate.shape[0] == 1:
            PredContour[Candidate[0,1].astype('int')] = Candidate[0,0] 
    
    # clip the padding of time
    PredContour = PredContour[range(half_ps, N-half_ps)]
    
    for k in range(len(PredContour)):
        if PredContour[k]>1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]
    
    Z = Z[:, range(half_ps, N-half_ps)]
    result = np.zeros([t.shape[0],2])
    result[:,0] = t/16000.0
    result[:,1] = PredContour
    return result

def findpeaks(x, th):
    # x is an input column vector
    M = x.shape[0]
    pre = x[1:M - 1] - x[0:M - 2]
    pre = np.sign(pre)
    
    post = x[1:M - 1] - x[2:]
    post = np.sign(post)

    mask = pre * post
    ext_mask = np.pad(mask,1)
    
    #pdata = x * ext_mask
    #pdata = pdata-np.tile(th*np.amax(pdata, axis=0),(M,1))
    #pks = np.where(pdata>0)
    #pks = pks[0]
    
    locs = np.where(ext_mask==1)
    locs = locs[0]
    return locs

def melody_extraction(filename1, filename2=None, mix_ratio=0.):
    
    # --- Args ---
    patch_size = 25
    th = 0.5
    modelname = 'model3_patch25'
    max_method = 'posterior'
    
    # --- Merge File ---
    audio = read_file(filename1)
    if filename2 is not None:
        aug_audio = read_file(filename2)
        
        # --- Combine Monophonic & Instrumental ---
        if len(audio) < len(aug_audio):
            randstart = np.random.randint(0, len(aug_audio)-len(audio)+1)
            audio = audio*(1-mix_ratio) + aug_audio[randstart:randstart+len(audio)]*mix_ratio
        else:
            shortage = len(audio)-len(aug_audio)
            randstart = np.random.randint(0, shortage+1)
            audio = audio*(1-mix_ratio) + np.pad(aug_audio, (randstart, shortage-randstart))*mix_ratio
    
    # --- Feature Extraction ---
    Z, t, CenFreq = melody_feature_extraction(audio)
    
    # --- Patch Extraction ---
    data, mapping, half_ps, N, Z = patch_extraction(Z, patch_size, th)
    
    # --- Pitch Prediction ---
    pred = patch_prediction(modelname, data, patch_size)
    result = contour_prediction(mapping, pred, N, half_ps, Z, t, CenFreq, max_method)

    return result