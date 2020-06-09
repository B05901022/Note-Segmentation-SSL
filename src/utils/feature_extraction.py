# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:51:19 2020

@author: Austin Hsu
"""

import scipy.io.wavfile
import scipy.signal
import scipy.fftpack
import numpy as np
import torchaudio.functional as F
import torch

def read_file(filename: str) -> (np.array, int):
    """Read files"""
    sample_rate, audio = scipy.io.wavfile.read(filename)
    audio = audio.astype(np.float32)
    if len(audio.shape)==2:
        audio = audio.mean(axis=-1)
    return audio, sample_rate

def torchaudio_STFT(x, fr, fs, Hop, h):
    """Can be further speedup with GPU"""
    window_size = h.size
    N = int(fs/fr) # 8000.0
    f = fs*np.linspace(0, 0.5, round(N/2), endpoint=True)
    t = np.arange(Hop, np.ceil(len(x)/Hop)*Hop, Hop)
    tfr = F.spectrogram(
            waveform = torch.from_numpy(x)[:-128], 
            pad = 0, 
            window = torch.from_numpy(h), 
            n_fft = int(fs-1),
            hop_length = Hop,
            win_length = window_size, 
            power = 1, 
            normalized = True).numpy()
    return tfr, f, t, N

def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/Hop)*Hop, Hop)
    N = int(fs/fr)
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
    f = 1/(q + 1e-8)
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

def gen_spectral_flux(S, invert=False, norm=True):
    flux = np.diff(S)
    first_col = np.zeros((S.shape[0],1))
    flux = np.hstack((first_col, flux))
    
    if invert:
        flux = flux * (-1.0)

    flux = np.where(flux < 0, 0.0, flux)

    if norm:
        flux = (flux - np.mean(flux)) / np.std(flux)

    return flux

def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    NumofLayer = np.size(g)

    tfr, f, t, N = STFT(x, fr, fs, Hop, h)
    tfr = np.power(tfr, g[0])
    tfr0 = tfr # original STFT
    ceps = np.zeros(tfr.shape)

    for gc in range(1, NumofLayer):
        if np.remainder(gc, 2) == 1:
            tc_idx = round(fs*tc) # 16
            ceps = np.real(np.fft.fft(tfr, axis=0))/np.sqrt(N)
            ceps = nonlinear_func(ceps, g[gc], tc_idx)
        else:
            fc_idx = round(fc/fr) # 40
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

def cfp_feature_extraction(audio: np.array, sample_rate: int) -> np.array:
    # --- Args ---
    fs = 16000.0 # sampling frequency
    Hop = 320 # hop size (in sample)
    h3 = scipy.signal.blackmanharris(743) # window size - 2048
    h2 = scipy.signal.blackmanharris(372) # window size - 1024
    h1 = scipy.signal.blackmanharris(186) # window size - 512
    fr = 2.0 # frequency resolution
    fc = 80.0 # the frequency of the lowest pitch
    tc = 1/1000.0 # the period of the highest pitch
    g = np.array([0.24, 0.6, 1])
    num_per_oct = 48 # Number of bins per octave
    
    #  --- Resample ---
    if sample_rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, sample_rate)
        
    # --- CFP Filterbank ---
    tfrL01, tfrLF1, tfrLQ1, f1, q1, t1, CenFreq1 = CFP_filterbank(audio, fr, fs, Hop, h1, fc, tc, g, num_per_oct)
    tfrL02, tfrLF2, tfrLQ2, f2, q2, t2, CenFreq2 = CFP_filterbank(audio, fr, fs, Hop, h2, fc, tc, g, num_per_oct)
    tfrL03, tfrLF3, tfrLQ3, f3, q3, t3, CenFreq3 = CFP_filterbank(audio, fr, fs, Hop, h3, fc, tc, g, num_per_oct)
    
    return tfrL01, tfrLF1, tfrLQ1, tfrL02, tfrLF2, tfrLQ2, tfrL03, tfrLF3, tfrLQ3

def full_feature_extraction(
        tfrL01, tfrLF1, tfrLQ1,
        tfrL02, tfrLF2, tfrLQ2,
        tfrL03, tfrLF3, tfrLQ3
        ):
    Z1 = tfrLF1 * tfrLQ1
    ZN1 = (Z1 - np.mean(Z1)) / np.std(Z1)
    Z2 = tfrLF2 * tfrLQ2
    ZN2 = (Z2 - np.mean(Z2)) / np.std(Z2)
    Z3 = tfrLF3 * tfrLQ3
    ZN3 = (Z3 - np.mean(Z3)) / np.std(Z3)
    SN1 = gen_spectral_flux(tfrL01, invert=False, norm=True)
    SN2 = gen_spectral_flux(tfrL02, invert=False, norm=True)
    SN3 = gen_spectral_flux(tfrL03, invert=False, norm=True)
    SIN1 = gen_spectral_flux(tfrL01, invert=True, norm=True)
    SIN2 = gen_spectral_flux(tfrL02, invert=True, norm=True)
    SIN3 = gen_spectral_flux(tfrL03, invert=True, norm=True)
    SN = np.concatenate((SN1, SN2, SN3), axis=0)
    SIN = np.concatenate((SIN1, SIN2, SIN3), axis=0)
    ZN = np.concatenate((ZN1, ZN2, ZN3), axis=0)
    SN_SIN_ZN = np.concatenate((SN, SIN, ZN), axis=0)
    return SN_SIN_ZN

def full_flow(audio, sample_rate):
    return full_feature_extraction(cfp_feature_extraction(audio, sample_rate))