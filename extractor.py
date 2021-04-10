import os
import wave
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank
from torch.nn.functional import adaptive_avg_pool2d
from random import sample

specpath = '.\\spec'
logmelpath = '.\\logmel'
MFCCpath = '.\\MFCC'
signalpath = '.\\signal'

nfft = 1024
sample_rate = 8000
nfilt = 128
ncep = 128
frame_length = 1024
frame_shift = 32
feature_shape = (50, 50)
satck_length = 20000


testset = ['subject_06', 'subject_08', 'subject_18', 'subject_20']
validset = ['subject_04', 'subject_12']

def readwave(path):
    f = wave.open(path,'rb')
    nchannel, sampwidth, framerate, nframes = f.getparams()[:4]
    strData = f.readframes(nframes)
    waveData = np.frombuffer(strData, dtype = np.int16)
    waveData = waveData / max(abs(waveData)) * 2
    f.close()
    return waveData

def pooling(data):
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    data = adaptive_avg_pool2d(data, feature_shape).squeeze().numpy()
    return data

def draw(data):
    plt.pcolor(data, cmap = 'jet')
    plt.show()
    plt.clf()

def extractfeature(wavedata):
    spec, freqs, ts, fig = plt.specgram(wavedata, NFFT = frame_length, \
        Fs = sample_rate,cmap = 'jet', noverlap = frame_length - frame_shift, window=np.hamming(M = frame_length))
    plt.clf()
    spec = np.log(spec)
    
    logmel= logfbank(wavedata, samplerate = sample_rate, winlen = frame_length / sample_rate, winstep = frame_shift / sample_rate, nfft = nfft, nfilt = nfilt)
    logmel = logmel.T

    MFCC = mfcc(wavedata, samplerate = sample_rate, winlen = frame_length / sample_rate, winstep = frame_shift / sample_rate, nfft = nfft, nfilt = nfilt, numcep = ncep, winfunc = np.hamming)
    MFCC = MFCC.T

    return spec, logmel, MFCC

def savefeature(data, path):
    f = open(path + '.pk', 'wb')
    pickle.dump(data, f)
    f.close()

def stack(data):
    temp = data.copy()
    while len(data) < satck_length:
        data = np.concatenate((data,temp))
    return data[: satck_length]

if __name__ == "__main__":

    if not os.path.exists(specpath):
        os.mkdir(specpath)
    if not os.path.exists(logmelpath):
        os.mkdir(logmelpath)
    if not os.path.exists(MFCCpath):
        os.mkdir(MFCCpath)
    
    trainpath = []
    trainlab = []
    validpath = []
    validlab = []
    testpath = []
    testlab = []
    U = []
    L = []
    for r, d, f in os.walk(signalpath):
        print(r)
        for name in d:
            dpath = os.path.join(specpath, r[len(signalpath) + 1: ], name)
            if not os.path.exists(dpath):
                os.mkdir(dpath)
            dpath = os.path.join(logmelpath, r[len(signalpath) + 1: ], name)
            if not os.path.exists(dpath):
                os.mkdir(dpath)
            dpath = os.path.join(MFCCpath, r[len(signalpath) + 1: ], name)
            if not os.path.exists(dpath):
                os.mkdir(dpath)
        
        for name in f:
            if name.endswith('.wav'):
                
                path, _ = os.path.splitext(name)
                path = os.path.join(r[len(signalpath) + 1: ], path + '.pk')
                lab = 1 if 'U' in r else 0
                istrain = True
                for p in testset:
                    if p in r:
                        testpath.append(path)
                        testlab.append(lab)
                        istrain = False
                        break
                for p in validset:
                    if p in r:
                        validpath.append(path)
                        validlab.append(lab)
                        istrain = False
                        break
                if istrain:
                    if lab == 1:
                        U.append((path, lab))
                    else:
                        L.append((path, lab))
                '''
                fpath = os.path.join(r, name)
                wavedata = stack(readwave(fpath))

                spec, logmel, MFCC = extractfeature(wavedata)
                
                spath, _ = os.path.splitext(os.path.join(specpath, r[len(signalpath) + 1: ], name))
                savefeature(spec, spath)

                spath, _ = os.path.splitext(os.path.join(logmelpath, r[len(signalpath) + 1: ], name))
                savefeature(logmel, spath)
                
                spath, _ = os.path.splitext(os.path.join(MFCCpath, r[len(signalpath) + 1: ], name))
                savefeature(MFCC, spath)
                '''
    aug = len(U) - len(L) # 1360 - 895 = 465
    L = L + sample(L, aug)
    path, lab = zip(*U)
    trainpath = trainpath + list(path)
    trainlab = trainlab + list(lab)
    path, lab = zip(*L)
    trainpath = trainpath + list(path)
    trainlab = trainlab + list(lab)
    pd.DataFrame({'path': trainpath, 'label': trainlab}).to_csv('train.csv')
    pd.DataFrame({'path': validpath, 'label': validlab}).to_csv('valid.csv')
    pd.DataFrame({'path': testpath, 'label': testlab}).to_csv('test.csv')
    
