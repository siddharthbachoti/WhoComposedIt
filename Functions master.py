#!/usr/bin/env python
# coding: utf-8

# ## This notebook contains all the functions used for data cleaning, data set generation, data augmentation, data visualization

# In[10]:


import numpy as np
import mido
from PIL import Image
from matplotlib import pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, MaxPooling1D, Conv2D, Conv1D, Flatten, Dropout, SpatialDropout1D, SimpleRNN, GRU, LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import random
import pandas as pd
import time
import pickle
import copy


# In[1]:


def midi_to_pickle(df,output,init,fin,inv_label_map,unavailable,num):#Specific to the Classical Archives HTML format

    '''
    Description: 
    Takes the raw data provided by classicalarchives.com, reads the MIDI files using Mido and stores in pickle files for later use.
    Also assigns the composer labels to each score.
    '''
    raw_data = []
    parent_directory = 'E:\\Classical_archives\\ClassicalArchives-MIDI-Collection\\m'

    for ind in df.index[init:fin]:
        name = df['composer_last_name'][ind]
        if name in inv_label_map:
            file_path = df['file_path'][ind]
            file_path = file_path.split('/')
            file_path = '\\'+ file_path[0] + '\\' + file_path[1]
            dirname = parent_directory + file_path
            num+=1
            temp = []
            temp.append(int(inv_label_map[name]))
            try:
                midfile = mido.MidiFile(dirname)

            except:
                unavailable.append(file_path)
            temp.append(midfile)
            raw_data.append(temp)

    picklename = output + '.pickle'#18
    with open(picklename, 'wb') as f:
        pickle.dump(raw_data, f)    
        
    time.sleep(3.5)


# In[2]:


def extract_raw_mid_files(parent_directory):#This is for when the pieces are grouped together in folders based on the composer name
    '''
    Description:
    This function does the same as midi_to_pickle but is for other data sets which are properly organized in composer-wise folder.
    '''
    label_map = {}
    for i, filename in enumerate(os.listdir(parent_directory)):
        label_map[str(i)] = filename

    inv_label_map = {v: k for k, v in label_map.items()}
    
    raw_data = []
    num = 0
    for filename in os.listdir(parent_directory):
        dirname = parent_directory + '\\' + filename
        for mid_song in os.listdir(dirname):
            num+=1
            temp=[]
            temp.append(int(inv_label_map[filename]))
            dir_of_mid_file = parent_directory + '\\' + filename + '\\' + mid_song
            midfile = mido.MidiFile(dir_of_mid_file)
            temp.append(midfile)
            raw_data.append(temp)    
    
    random.shuffle(raw_data)
    
    return raw_data, num, label_map


# In[16]:


def midi_to_p_roll(mid,Nyquist_rate,sample_duration,pitch_range):
    
    '''
    Description:
    Converts a MIDI file into a piano roll of required time length and pitch range.
    
    **Algorithm was inspired by the method adopted by Jain et al. (http://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26583519.pdf)
    '''

    piano_size = pitch_range[1] - pitch_range[0]
    p_roll = np.zeros([piano_size+1, Nyquist_rate * sample_duration])
    track = mido.merge_tracks(mid.tracks)

    current_time = 0
    current_position = 0
    on_notes = np.zeros(piano_size+1)
    tempo = 0
    
    for msg in track:

        if msg.time > 0:
            delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        else:
            delta = 0
        if hasattr(msg, "note"):
            if msg.type == "note_on":
                if pitch_range[0] <= msg.note <= pitch_range[1]:
                    on_notes[msg.note-pitch_range[0]] = msg.velocity
            else:
                if pitch_range[0] <= msg.note <= pitch_range[1]:
                    on_notes[msg.note-pitch_range[0]] = 0
        last_time = current_time
        current_time += delta

        if current_time > sample_duration:
            break

        new_position = np.floor(current_time * Nyquist_rate).astype(int)

        if new_position > current_position:
            new_position = np.floor(current_time * Nyquist_rate).astype(int)
            block = np.tile(on_notes.reshape(piano_size+1, 1), new_position - current_position)
            p_roll[:, current_position:new_position] = block
            current_position = new_position
            
        if hasattr(msg, "tempo"):
            tempo = msg.tempo

    return p_roll


# In[17]:


def generate_data_sets(raw_data, ratio):
    
    '''
    Description:
    Generates the datasets used for deep learning experiments and divides them in the required train-test ratio.
    '''
    #Dataset division ratio must be given in the order train,test
    labels = []
    piano_rolls = []
    
    for i in raw_data:
        labels.append(i[0])
        piano_rolls.append(midi_to_p_roll(i[1],10,70,(30,100)))
    
    num_train = int(ratio[0]*len(labels))
    
    training_data = piano_rolls[0:num_train]
    test_data = piano_rolls[num_train:]
    training_labels = labels[0:num_train]
    test_labels = labels[num_train:]
    
    return (np.array(training_data),np.array(training_labels)), (np.array(test_data),np.array(test_labels))


# In[25]:


def generate_p_roll_plot(roll,t1,t2,min_note,max_note):
    
    '''
    Description:
    Generates visual images of a piano roll
    '''
    plt.axis([t1,t2,min_note,max_note])
    plt.imshow(roll, cmap='Greys')


# In[18]:


def make_time_series(data):
    '''
    Description:
    Converts a piano roll image into time series in order to be fed into recurrent layers
    '''
    data1 = []
    for i in data:
        data1.append(np.transpose(i))
    return np.array(data1)


# In[2]:


def remove_dynamics(arr):
    '''
    Description:
    Equalises the veloocities of all notes by removing any notion of relative loudness or volume
    '''
    for i in np.transpose(np.nonzero(arr)):
        arr[i[0]][i[1]] = 1
    return arr


# In[3]:


def transpose_scale(arr,num):
    
    '''
    Description:
    (Data augmentation) Takes the piano roll and transposes the scale of the piece by 'num' units
    '''
    
    arr2 = copy.deepcopy(arr)
    arr2 = np.roll(arr2,-num,axis = 0)
    if int(num/abs(num))==1:
        for i in range(num):
            arr2[-(i+1)] = np.zeros(len(arr2[0]))
    else:
        for i in range(abs(num)):
            arr2[i] = np.zeros(len(arr2[0]))
    return arr2


# In[ ]:


def translate_time(arr,num):
    
    '''
    Description:
    (Data augmentation) Performs a translation of the direction of flow of time
    '''
    
    arr2 = copy.deepcopy(arr)
    arr2 = np.transpose(arr2)
    arr2 = np.roll(arr2,-num,axis = 0)
    if int(num/abs(num))==1:
        for i in range(num):
            arr2[-(i+1)] = np.zeros(len(arr2[0]))
    else:
        for i in range(abs(num)):
            arr2[i] = np.zeros(len(arr2[0]))
            
    return np.transpose(arr2)

