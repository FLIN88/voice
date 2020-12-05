import os
import matplotlib.pyplot as plt
import wave
import numpy as np
from scipy import signal
from scipy.fftpack import dct

N = 1024; #  FFT点数
P = 128; # 滤波器数量
L = 64; # DCT阶数
frame_length = 512; # 帧长
frame_shift = 32; # 帧移
tot_wavefile_cnt=0 #统计 .wav 文件总数

lifts=[]
for n in range(1, L + 1):
    lift = 1 + 6 * np.sin(np.pi * n / L)
    lifts.append(lift)

# 构造melbank
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (8000 / 2) / 700))
mel_points = np.linspace(low_freq_mel, high_freq_mel, P + 2)
hz_points = (700 * (10 ** (mel_points / 2595) - 1)) # 把 Mel 变成 Hz
fbank = np.zeros((P, int(np.floor(N / 2 + 1))))
bin = np.floor((N + 1) * hz_points / 8000) # 把Hz换成df下的编号
for m in range(1, P + 1): #梅尔滤波器信号的采集
    f_m_minus = int(bin[m - 1])  # left
    f_m = int(bin[m])       # center 梅尔滤波器的三级峰
    f_m_plus = int(bin[m + 1])  # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
fbank = fbank.T

def enframe(wave_data, frame_length, frame_shift, wind):
    '''将音频信号转化为帧。
    参数含义：
    wave_data:原始音频型号
    frame_length:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    frame_shift:相邻帧的间隔（同上定义）
    nf:帧数
    '''
    wlen = len(wave_data) #信号总长度
    if wlen <= frame_length: #若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else: #否则，计算帧的总长度
        nf = int(np.ceil((1.0 * wlen - frame_length + frame_shift) / frame_shift))
    pad_length = int((nf - 1)*frame_shift + frame_length) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length - wlen,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((wave_data, zeros)) #填补后的信号记为pad_signal
    indices = np.tile(np.arange(0,frame_length), (nf,1)) + np.tile(np.arange(0, nf * frame_shift, frame_shift), (frame_length,1)).T #相当于对所有帧的时间点进行抽取，得到nf*frame_length长度的矩阵
    indices = np.array(indices, dtype = np.int32) #将indices转化为矩阵
    frames = pad_signal[indices] #得到帧信号
    win = np.tile(wind, (nf, 1)) #window窗函数，这里默认取1
    return frames * win  #返回帧信号矩阵

def output(vec, f):
    out=vec.reshape(1,-1)[0]
    #print(len(out))
    for v in out:
        print(v, end = ' ', file = f)
    print(file = f)


def deal(inpath, outpath, lab):
    global frame_length, frame_shift, lifts, minf
    # 先把音频文件读入，波形图已数组形式存储
    f = wave.open(inpath,'rb')
    nchannel, sampwidth, framerate, nframes = f.getparams()[:4]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData, dtype = np.int16)
    waveData = waveData / max(abs(waveData)) * 2
    f.close()

    path, name = os.path.split(outpath)
    prefix, sufix = name.split('.')
    
    
    # 语谱图
    spectrum, freqs, ts, fig = plt.specgram(waveData, NFFT = frame_length, pad_to = N, \
        Fs = framerate,cmap = 'jet', noverlap = frame_length - frame_shift,window=np.hamming(M = frame_length))
    plt.axis('off')
    plt.savefig(path + '/spec_' + prefix + '.png', bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
    
    #print('spectrum.shape',spectrum.shape)
    #spectrum = np.log10(spectrum) * 10
    #spectrum = spectrum.reshape(1,-1)[0]
    
    # MFCC
    
    frames = enframe(waveData, frame_length, frame_shift, np.hamming(frame_length))
    nf = frames.shape[0] - 1
    #MFCC = np.zeros((L, nf))
    logMel = np.zeros((P, nf))
    
    for i in range(nf):
        y = frames[i, :]
        yf = np.abs(np.fft.fft(y, n = N))
        yf = yf ** 2    # 从FFT结果得到能量
        
        filter_banks = np.dot(yf[0:N // 2 + 1], fbank) # 得到mel能量
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) # 数值稳定性
        filter_banks = 10 * np.log10(filter_banks) # dB
        logMel[:, i] = filter_banks
        '''
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8) # 把权值下移平均值
        
        c2 = dct(filter_banks, type=2, axis=-1, norm='ortho')[ 1 : (L + 1)] # Keep 2-13
        c2 *= lifts
        MFCC[:, i] = c2
        '''
    
    # 画logMel
    
    plt.pcolor(logMel, cmap = 'jet')
    plt.axis('off')
    plt.savefig(path + '/' + str(lab) + '_logMel_' + prefix + '.png', bbox_inches = 'tight',pad_inches = 0, dpi =20)
    plt.clf()
    '''
    #画MFCC
    plt.pcolor(MFCC,cmap = 'jet')
    plt.axis('off')
    plt.savefig(path + '/MFCC_' + prefix + '.png', bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
    '''
def findwav(inpath, outpath, lab):
    '''
    扫描当前文件夹 inpath 的文件 f
    如果是wav文件就处理并输出同名文件到 outpath
    如果是文件夹就在 outpath 建立同名文件夹递归处理
    '''
    global tot_wavefile_cnt
    filelist = os.listdir(inpath)
    for f in filelist:
        inname = inpath + '/' + f
        outname = outpath + '/' + f
        if os.path.isdir(inname):
            if f.startswith('U') or lab == 1:
                lab = 1
            else:
                lab = 0 
            if not os.path.exists(outname):
                os.mkdir(outname)
            findwav(inname, outname, lab)
        elif f.endswith('.wav'):
            deal(inname, outname, lab)
            tot_wavefile_cnt+=1
            print(tot_wavefile_cnt,"wavefiles finished...")


if __name__ == "__main__":
    rootpath = './signal'
    newfolder = './images'

    if not os.path.exists(newfolder):
        os.mkdir(newfolder)

    findwav(rootpath, newfolder, 0)

    print("Total",tot_wavefile_cnt,"wavefiles dealt!")

    #deal(r'./test.wav','./test.wav', 0, fvec, flab)
