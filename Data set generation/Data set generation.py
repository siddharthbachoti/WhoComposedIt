#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import mido
from PIL import Image
from matplotlib import pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.utils import to_categorical
import random
import pandas as pd
import time
import pickle
import copy


# In[ ]:





# In[2]:


def midi_to_pickle(df,output,init,fin,inv_label_map,unavailable,num):#Specific to the Classical Archives HTML format

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
                temp.append(midfile)
                raw_data.append(temp)
                
            except:
                unavailable.append(file_path)


    picklename = output + '.pickle'#18
    with open(picklename, 'wb') as f:
        pickle.dump(raw_data, f)    
        
    time.sleep(3.5)


# In[3]:


def midi_to_p_roll(mid,Nyquist_rate,sample_duration):

    p_roll = np.zeros([128, Nyquist_rate * sample_duration])
    track = mido.merge_tracks(mid.tracks)

    current_time = 0
    current_position = 0
    on_notes = np.zeros(128)
    tempo = 0
    
    for msg in track:

        if msg.time > 0:
            delta = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        else:
            delta = 0
        if hasattr(msg, "note"):
            if msg.type == "note_on":
                on_notes[msg.note] = msg.velocity
            else:
                on_notes[msg.note] = 0
        last_time = current_time
        current_time += delta

        if current_time > sample_duration:
            break

        new_position = np.floor(current_time * Nyquist_rate).astype(int)

        if new_position > current_position:
            new_position = np.floor(current_time * Nyquist_rate).astype(int)
            block = np.tile(on_notes.reshape(
                128, 1), new_position - current_position)
            p_roll[:, current_position:new_position] = block
            current_position = new_position
            
        if hasattr(msg, "tempo"):
            tempo = msg.tempo

    return p_roll


# # Converting MIDI files into Piano rolls and outputting into pickle files

# In[13]:


for i in range(1):
    picklename = 'data' + str(i+1) + '.pickle'
    with open(picklename, 'rb') as f:
        mid = pickle.load(f)
    for j in mid:    
        j[1] = midi_to_p_roll(j[1], 10, 60)
    
    p_roll_name = 'p_roll_test' + str(i+1) + '.pickle'
    with open(p_roll_name, 'wb') as f:
        pickle.dump(mid, f)    
    


# In[16]:


with open(p_roll_name, 'rb') as f:
    new = pickle.load(f)


# # Creating MIDI pickle files using midi_to_pickle

# In[43]:


df = pd.read_excel('E:\Classical_archives\ClassicalArchives-MIDI-Collection\midi_catalog.xls')
for ind in df.index:
    if df['composer_last_name'][ind] == "Deleted":
        temp = df['category_chain'][ind]
        temp = temp.split(',')[0]
        df['composer_last_name'][ind] = temp
        
dict1 = {}

for i in list(df['composer_last_name']):

    if i in dict1:
        dict1[i]+=1
    else:
        dict1[i]=1
        
lst = list(dict1.items())
lst = sorted(lst, key=lambda x: x[1], reverse=True)

label_map = {}
for i, x in enumerate(lst[0:20]):
    label_map[i] = x[0]

inv_label_map = {v: k for k, v in label_map.items()}

parent_directory = 'E:\\Classical_archives\\ClassicalArchives-MIDI-Collection\\m'


# In[47]:


i=0
flag = 0
init = 0
fin = 300
num=0
unavailable = []


# In[48]:


while flag==0: 
    
    i+=1

    output = 'midi_' + str(i)
    midi_to_pickle(df,output,init,fin,inv_label_map,unavailable,num)
    
    init+=300
    fin+=300
    
    if fin>len(df):
        flag =1


# In[52]:


with open('midi_68.pickle', 'rb') as f:
    new = pickle.load(f) 


# In[ ]:


num = 0
for i in range(1,73):
    name = 'midi_' + str(i) + '.pickle'
    with open(name, 'rb') as f:
        new = pickle.load(f)
    rolls = copy.deepcopy(new)
    for j in rolls:
        j[1] = midi_to_p_roll(j[1],10,154)
        
    roll_name = 'p_roll_154_' + str(i) + '.pickle'
        
    with open(roll_name, 'wb') as f:
        pickle.dump(rolls, f)      


# # Creating data sets for trinary classification between Bach, Beethoven and Mozart

# In[2]:



df = pd.read_excel('E:\Classical_archives\ClassicalArchives-MIDI-Collection\midi_catalog.xls')
for ind in df.index:
    if df['composer_last_name'][ind] == "Deleted":
        temp = df['category_chain'][ind]
        temp = temp.split(',')[0]
        df['composer_last_name'][ind] = temp
        
dict1 = {}

for i in list(df['composer_last_name']):

    if i in dict1:
        dict1[i]+=1
    else:
        dict1[i]=1
        
lst = list(dict1.items())
lst = sorted(lst, key=lambda x: x[1], reverse=True)

label_map = {}
for i, x in enumerate(lst[0:4]):
    label_map[i] = x[0]

inv_label_map = {v: k for k, v in label_map.items()}

parent_directory = 'E:\\Classical_archives\\ClassicalArchives-MIDI-Collection\\m'


# In[7]:


i=0
flag = 0
init = 0
fin = 500
num=0
unavailable = []
bach = []
mozart = []
beethoven = []
##Enter more composer lists if you want.##


# ## Creation of MIDI Pickle files

# In[8]:


label_map


# In[9]:


for ind in df.index:
    
    if df['composer_last_name'][ind] == 'Beethoven' and len(beethoven)<=500:
        name = 'Beethoven'
        temp = []
        file_path = df['file_path'][ind]
        file_path = file_path.split('/')
        file_path = '\\'+ file_path[0] + '\\' + file_path[1]
        dirname = parent_directory + file_path
        temp.append(int(inv_label_map[name]))
        
        try:
            midfile = mido.MidiFile(dirname)
            temp.append(midfile)
            beethoven.append(temp)
            
        except:
            unavailable.append(file_path)
            
#     if df['composer_last_name'][ind] == 'Bach' and len(bach)<=500:
#         name = 'Bach'
#         temp = []
#         file_path = df['file_path'][ind]
#         file_path = file_path.split('/')
#         file_path = '\\'+ file_path[0] + '\\' + file_path[1]
#         dirname = parent_directory + file_path
#         temp.append(int(inv_label_map[name]))
        
#         try:
#             midfile = mido.MidiFile(dirname)
#             temp.append(midfile)
#             bach.append(temp)
            
#         except:
#             unavailable.append(file_path)
            
#     if df['composer_last_name'][ind] == 'Mozart' and len(mozart)<=500:
#         name = 'Mozart'
#         temp = []
#         file_path = df['file_path'][ind]
#         file_path = file_path.split('/')
#         file_path = '\\'+ file_path[0] + '\\' + file_path[1]
#         dirname = parent_directory + file_path
#         temp.append(int(inv_label_map[name]))
        
#         try:
#             midfile = mido.MidiFile(dirname)
#             temp.append(midfile)
#             mozart.append(temp)
            
#         except:
#             unavailable.append(file_path)
        


# In[10]:


len(beethoven)


# In[11]:


with open('beethoven.pickle', 'wb') as f:
    pickle.dump(beethoven, f)


# In[29]:


with open('bach.pickle', 'wb') as f:
    pickle.dump(bach, f)
        
with open('mozart.pickle', 'wb') as f:
    pickle.dump(mozart, f)


# ## Creation of piano rolls

# In[24]:


A = ['mozart','bach']

for name in A:
    picklefile = name + '.pickle'
    
    with open(picklefile, 'rb') as f:
        new = pickle.load(f)
    rolls = copy.deepcopy(new)
    for j in rolls:
        j[1] = midi_to_p_roll(j[1],10,154)
        
    roll_name = 'p_roll_154_' + name + '.pickle'
        
    with open(roll_name, 'wb') as f:
        pickle.dump(rolls, f)      


# In[25]:


inv_label_map


# In[ ]:




