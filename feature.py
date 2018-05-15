# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:48:24 2017

@author: flyaway
"""
import librosa
from  librosa import feature
import scipy.io.wavfile as wav
import numpy 
def get_feature_from_python_speech_features(wave_name):
    from python_speech_features import logfbank
    from python_speech_features import mfcc
    from python_speech_features import delta
    from python_speech_features import fbank
    from python_speech_features import ssc
    import scipy.io.wavfile as wav
    import numpy 
    (rate,sig) = wav.read(wave_name)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    d_d_mfcc_feat = delta(d_mfcc_feat,2)
    fbank_feat,energy = fbank(sig,rate)
    logfbank_feat = logfbank(sig,rate)
    centroids = ssc(sig,rate)
    feat = numpy.hstack((mfcc_feat,d_mfcc_feat,d_d_mfcc_feat,logfbank_feat,centroids))
    return feat.T#一行代表一帧的特征


def get_feature_from_librosa(wave_name,window):
    #print wave_name
    (rate,sig) = wav.read(wave_name)

    chroma_stft_feat = feature.chroma_stft(sig,rate,n_fft=window, hop_length=window/2)
    #print chroma_stft_feat.shape
    mfcc_feat = feature.mfcc(y=sig,sr=rate,n_mfcc=13,hop_length=window/2)
    mfcc_feat = mfcc_feat[1:,:]
    #print mfcc_feat.shape
    d_mfcc_feat = feature.delta(mfcc_feat)
    #print d_mfcc_feat.shape
    d_d_mfcc_feat = feature.delta(d_mfcc_feat)
    #print d_d_mfcc_feat.shape
    zero_crossing_rate_feat = feature.zero_crossing_rate(sig,frame_length=window, hop_length=window/2)
    #print zero_crossing_rate_feat.shape
    
    S = librosa.magphase(librosa.stft(sig, hop_length=window/2, win_length=window, window='hann'))[0]
    rmse_feat = feature.rmse(S=S)
    #print rmse_feat.shape
    
    centroid_feat = feature.spectral_centroid(sig,rate,n_fft=window, hop_length=window/2)
    #print centroid_feat.shape
    
    bandwith_feat = feature.spectral_bandwidth(sig,rate,n_fft=window,hop_length=window/2)
    #print bandwith_feat.shape
    
    contrast_feat = feature.spectral_contrast(sig,rate,n_fft=window,hop_length=window/2)
    #print contrast_feat.shape
    rolloff_feat = feature.spectral_rolloff(sig,rate,n_fft=window,hop_length=window/2)#计算滚降频率
    #print rolloff_feat.shape
    
    poly_feat = feature.poly_features(sig,rate,n_fft=window,hop_length=window/2)#拟合一个n阶多项式到谱图列的系数。
    #print poly_feat.shape
#==============================================================================
#     print(chroma_stft_feat.shape)
#     #print(corr_feat.shape)
#     print(mfcc_feat.shape)
#     print(d_mfcc_feat.shape)
#     print(d_d_mfcc_feat.shape)
#     print(zero_crossing_rate_feat.shape)
#     print(rmse_feat.shape)
#     print(centroid_feat.shape)
#     print(bandwith_feat.shape)
#     print(contrast_feat.shape)
#     print(rolloff_feat.shape)
#     print(poly_feat.shape)
#==============================================================================
    feat = numpy.hstack((chroma_stft_feat.T,mfcc_feat.T,d_mfcc_feat.T,d_d_mfcc_feat.T,
                         zero_crossing_rate_feat.T,rmse_feat.T,centroid_feat.T,bandwith_feat.T,
                         contrast_feat.T,rolloff_feat.T,poly_feat.T))
    feat = feat.T
    return feat#一行代表一帧的特征

def get_mult_feature(wave_name,window):
    (rate,sig) = wav.read(wave_name)    # librosa.load()
    #
    #power = librosa.amplitude_to_db(librosa.stft(sig,n_fft = window,hop_length=window/2), ref=numpy.max)
    
    power = numpy.abs(librosa.stft(sig,n_fft = window,hop_length=window/2))
    _min = numpy.min(power)
    _max = numpy.max(power)
    power = (power - _min) / (_max - _min)

        
    print power.shape
    
    mels = feature.melspectrogram(sig,rate,n_fft = window, hop_length=window/2,n_mels = 257)
    _min = numpy.min(mels)
    _max = numpy.max(mels)
    mels = (mels - _min) / (_max - _min)

    print mels.shape
    
    if power.shape[1] < 250:
        zero = numpy.zeros((power.shape[0],250 - power.shape[1]))
        power = numpy.hstack((power,zero))
        mels = numpy.hstack((mels,zero))
    else:
        power = power[:,0:250]
        mels = mels[:,0:250]
    power = power.T
    mels = mels.T
    feat = numpy.asarray([power,mels])
    return feat
    
import glob
import os
def read_wav(dir_name,window,mult = False):
    #print(dir_name)
    file_num = sum([len(x) for _, _, x in os.walk(os.path.dirname(dir_name))]) #read the number of the file in the Folder
    print(file_num)
    feat = []
    index = 0
    maxlen = 1
    
    for file in glob.glob(dir_name + '*.wav'):
        #print(file)
        #print(index)
        #print(file)
        index += 1
        if mult  == False:
            feature1= get_feature_from_librosa(file,window)
        else:
            feature1 = get_mult_feature(file,window)
        #print feature1.shape
        maxlen = max(maxlen,feature1.shape[1])
        feat.append(feature1)
        #print(len(feat))
    return feat,maxlen