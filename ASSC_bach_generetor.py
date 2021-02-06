import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os
import keras
from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D,concatenate, SpatialDropout1D, TimeDistributed, Bidirectional, LSTM
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import IPython
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras_contrib.layers import CRF
from keras.layers import Convolution2D, MaxPool2D, GlobalMaxPool2D, GlobalAveragePooling2D, SpatialDropout2D
#from models import get_model_cnn
import numpy as np
#from utils import gen, chunker, WINDOW_SIZE, rescale_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, classification_report
from glob import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
import numpy
from sklearn.preprocessing import StandardScaler
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
#!pip install keras-self-attention
#from keras_self_attention import SeqSelfAttention
#!pip inimport pandas as pd 
from scipy.signal import argrelextrema
#from kutilities.layers import AttentionWithContext , MeanOverTime
#!pip install keras-tcn
#!pip install keract
#!pip install catboost
#!pip install imblearn
from scipy.signal import butter, lfilter
import h5py
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
from imblearn.under_sampling import RandomUnderSampler
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
def scaling_filtering(X):
       
        for i in range(len(X)):
            
            d=butter_bandpass_filter(X[i].reshape(3000),.1 ,37, 100, order=5)
            X[i]=d.reshape(3000,1)
            X[i]=(X[i]-numpy.mean(X[i]))/numpy.std(X[i])
        return  X    
        
def bach_generator_aug(dict_files):
    while True:
      
        for i in range(30):
                j=0
                ind=[]
                WINDOW_SIZE1 = 100
                WINDOW_SIZE2 = 400
  
                record_name1 = random.choice(list(dict_files.keys()))
                batch_data1 = dict_files[record_name1]
                all_rows1 = batch_data1['x']


                record_name2 = random.choice(list(dict_files.keys()))
                batch_data2 = dict_files[record_name2]
                all_rows2 = batch_data2['x']


                start_index = random.choice(range(all_rows1.shape[0]-WINDOW_SIZE1))
                X1 = all_rows1[start_index:start_index+WINDOW_SIZE1, ...]
                Y1 = batch_data1['y'][start_index:start_index+WINDOW_SIZE1]



                start_index = random.choice(range(all_rows2.shape[0]-WINDOW_SIZE2))
                X2 = all_rows2[start_index:start_index+WINDOW_SIZE2, ...]
                Y2 = batch_data2['y'][start_index:start_index+WINDOW_SIZE2]
                for k in range(5):
                
                      for i in range(len(Y2)):
                            if  j==100:
                                 break
                            if Y1[j]==Y2[i]:
                                ind+=[i]
                                j=j+1
                         
                if  len(ind)==100:
                    break
        
        X=X2[ind]
        X =scaling_filtering(X)
        X = np.expand_dims(X, 0)
        
        X1 =rescale_array2(X1)
        X1 = np.expand_dims(X1, 0)
        
        Y1 = np.expand_dims(Y1, -1)
        Y1 = np.expand_dims(Y1, 0)
        yield [X,X1],Y1    
                    
def chunker(seq, size=WINDOW_SIZE1):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
def balance_generetor(dict_files):
    j=0
    while True:
        for i in range(len(dict_files)):
                record_name = random.choice(list(dict_files.keys()))
                batch_data = dict_files[record_name]
                all_rows = batch_data['x']
                all_rows=all_rows.reshape(len(all_rows),3000)
                y=batch_data['y']
                rus = RandomUnderSampler(random_state=0, replacement=True)
                X_resampled, y_resampled = rus.fit_resample(all_rows, y)
                X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=0)
                X_resampled=X_resampled.reshape(len(X_resampled),3000,1)
                if len(y_resampled)>100:
                    break

        for i in range(1):
            start_index = random.choice(range(X_resampled.shape[0]-WINDOW_SIZE))

            X1 = X_resampled[start_index:start_index+WINDOW_SIZE, ...]
            Y1 = y_resampled[start_index:start_index+WINDOW_SIZE]
            X1 =scaling_filtering(X1)
            X1 = np.expand_dims(X1, 0)
            Y1 = np.expand_dims(Y1, -1)
            Y1 = np.expand_dims(Y1, 0)
            yield [X1,X1],Y1
def bach_generetor(dict_files):
    while True:
  
        record_name = random.choice(list(dict_files.keys()))
        #all_rows = np.load(record_name)#+'.npy'
        batch_data = dict_files[record_name]
        all_rows = batch_data['x']
        start_index = random.choice(range(all_rows.shape[0]-WINDOW_SIZE))
        X = all_rows[start_index:start_index+WINDOW_SIZE, ...]
        Y1 = batch_data['y'][start_index:start_index+WINDOW_SIZE]
        X =rescale_array2(X)
        X = np.expand_dims(X, 0)
        Y1 = np.expand_dims(Y1, -1)
        Y1 = np.expand_dims(Y1, 0)
        yield [X,X],Y1 
        
