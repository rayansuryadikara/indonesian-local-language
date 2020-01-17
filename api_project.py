# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:28:26 2019

@author: T'Chala
"""

#Check if Indonesian local languages could be recognized by deep learning speech classification.
#Audio data source is from bible.is, which is Bible audio book available for almost 700 language with 
#---similar order for each languages
 
import os
import random as rn
import subprocess
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import pickle
from keras.layers import Dense
from keras import Input
from keras.engine import Model
from keras.utils import to_categorical
from keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, \
    LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D

#DATA_DIR = '\\Audio Processing and Indexing\\API Project\\bkaba'
#random_file = rn.choice(os.listdir(os.path.abspath('') + DATA_DIR))
# Function to rename multiple files 

#for filename in os.listdir(os.path.abspath('') + DATA_DIR):
#    print(filename[:-5] + '_' + filename[-5:] )
#    break
#def main(): 
##    i = 0
#      
#    for filename in os.listdir(os.path.abspath('') + DATA_DIR):
#        newfilename = filename[:-6] + '_' + filename[-6:] 
#          
#        # rename() function will 
#        # rename all the files 
#        os.rename(os.path.join(os.path.abspath('') + DATA_DIR, filename), os.path.join(os.path.abspath('') + DATA_DIR, newfilename))
##        i += 1
#  
## Driver Code 
#if __name__ == '__main__': 
#      
#    # Calling main() function 
#    main() 

os.chdir("C:/Users/T'Chala/Documents/Rayan's Document/Leiden/Literatures/Audio Processing and Indexing/API Project/Dataset")

count = -2
count2 = 0
count3 = 0
count = count + 2
count2 = count2 + 2
count3 = count3 + 1
ffmpeg = 'ffmpeg -i B27___01_Wahyu_______LBWLAIN1DA.mp3 -c copy -ss %d -to %d B27_01_Wahyu_Tolaki_%d.wav' % (count, count2, count3)
subprocess.call(ffmpeg, shell=True)

DATA_DIR = '\\Audio Processing and Indexing\\API Project\\Dataset'
random_file = rn.choice(os.listdir(os.path.abspath('') + DATA_DIR))

wav, sr = librosa.load(os.path.abspath('') + DATA_DIR + '\\' + random_file)
print('sr:', sr)
print('wav shape:', wav.shape)
print('length:', wav.shape[0]/float(sr), 'secs')

plt.plot(wav)

D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')

chapters = ['Matius_01', 'Matius_02', 'Markus_01', 'Markus_02', 'Lukas_01', 'Lukas_02', 'Yohanes_01', 'KisahRasul_01', 
            'Roma_01', '1Korintus_01', '2Korintus_01', 'Galatia_01', 'Efesus_01', 'Filipe_01', 'Kolose_01', '1Tesalonika_01',
            '2Tesalonika_01', '1Timotius_01', '2Timotius_01', 'Titus_01', 'Filemon_01', 'Ibrani_01', 'Yakobus_01', '1Petrus_01',
            '2Petrus_01', '1Yohanes_01', '2Yohanes_01', '3Yohanes_01', 'Yudas_01', 'Wahyu_01']

rn.seed(6)
test_chapter = rn.choices(chapters, k = 5)

train_X = []
train_spectrograms = []
train_mel_spectrograms = []
train_mfccs = []
train_y = []

test_X = []
test_spectrograms = []
test_mel_spectrograms = []
test_mfccs = []
test_y = []

audio_data = []
audio_spectrograms = []
audio_mel_spectrograms = []
audio_mfccs = []
audio_labels = []
audio_chapters = []

pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

count = 0

start = time.time()
for fname in os.listdir(os.path.abspath('') + DATA_DIR):
    try:
        if '.wav' not in fname or 'dima' in fname:
            continue
        struct = fname.split('_')
        chapter = struct[2] + '_' + struct[1] 
        language = struct[3]
#        wav, sr = librosa.load(os.path.abspath('') + DATA_DIR + '\\' + fname)
##        padded_x = pad1d(wav, 44100)
#        spectrogram = np.abs(librosa.stft(wav))
#        padded_spectogram = pad2d(spectrogram,80)
#
#        mel_spectrogram = librosa.feature.melspectrogram(wav)
#        padded_mel_spectrogram = pad2d(mel_spectrogram,80)
#        
#        mfcc = librosa.feature.mfcc(wav)
#        padded_mfcc = pad2d(mfcc,80)

#        if chapter in test_chapter: 
#            test_X.append(padded_x)
#            test_spectrograms.append(padded_spectogram)
#            test_mel_spectrograms.append(padded_mel_spectrogram)
#            test_mfccs.append(padded_mfcc)
#            test_y.append(language)
#        else:
#            train_X.append(padded_x)
#            train_spectrograms.append(padded_spectogram)
#            train_mel_spectrograms.append(padded_mel_spectrogram)
#            train_mfccs.append(padded_mfcc)
#            train_y.append(language)
#        audio_data.append(padded_x)
#        audio_spectrograms.append(padded_spectogram)
#        audio_mel_spectrograms.append(padded_mel_spectrogram)
#        audio_mfccs.append(padded_mfcc)
#        audio_labels.append(language)
        audio_chapters.append(chapter)
        
        print("File", count+1, "done!")
        count = count + 1
    except Exception as e:
        print(fname, e)
        raise
end = time.time()
print(end-start)

#for i in range(len(audio_labels)):
#    if audio_labels[i] == 'Karo':
#        audio_labels[i] = 0
#    elif audio_labels[i] == 'Sunda':
#        audio_labels[i] = 1
#    elif audio_labels[i] == 'Tolaki':
#        audio_labels[i] = 2   

#with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\train_X.data', 'wb') as fp:
#    pickle.dump(train_X, fp)
#fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\audio_spectrograms.data', 'wb') as fp:
    pickle.dump(audio_spectrograms, fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\audio_mel_spectrograms.data', 'wb') as fp:
    pickle.dump(audio_mel_spectrograms, fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\audio_mfccs.data', 'wb') as fp:
    pickle.dump(audio_mfccs, fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\audio_labels.data', 'wb') as fp:
    pickle.dump(audio_labels, fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\audio_chapters.data', 'wb') as fp:
    pickle.dump(audio_chapters, fp)
fp.close()

#with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\audio_X.data', 'wb') as fp:
#    pickle.dump(test_X, fp)
#fp.close()
#with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_spectrograms.data', 'wb') as fp:
#    pickle.dump(test_spectrograms, fp)
#fp.close()
#with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_mel_spectrograms.data', 'wb') as fp:
#    pickle.dump(test_mel_spectrograms, fp)
#fp.close()
#with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_mfccs.data', 'wb') as fp:
#    pickle.dump(test_mfccs, fp)
#fp.close()
#with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_y.data', 'wb') as fp:
#    pickle.dump(test_y, fp)
#fp.close()

#        complistnew = json.load(fp)

with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\train_X.data', 'rb') as fp:
    train_X = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\train_spectrograms.data', 'rb') as fp:
    train_spectrograms = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\train_mel_spectrograms.data', 'rb') as fp:
    train_mel_spectrograms = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\train_mfccs.data', 'rb') as fp:
    train_mfccs = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\train_y.data', 'rb') as fp:
    train_y = pickle.load(fp)
fp.close()

with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_X.data', 'rb') as fp:
    test_X = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_spectrograms.data', 'rb') as fp:
    test_spectrograms = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_mel_spectrograms.data', 'rb') as fp:
    test_mel_spectrograms = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_mfccs.data', 'rb') as fp:
    test_mfccs = pickle.load(fp)
fp.close()
with open(os.path.abspath('') + '\\Audio Processing and Indexing\\API Project\\test_y.data', 'rb') as fp:
    test_y = pickle.load(fp)
fp.close()   
    
d = {'spectrogram': audio_spectrograms,
     'mel_spectrogram': audio_mel_spectrograms,
     'mfccs': audio_mfccs,
     'labels' : audio_labels,
     'chapters': audio_chapters}

df = pd.DataFrame(data=d)

train_X = np.vstack(train_X)
train_spectrograms = np.array(train_spectrograms)
train_mel_spectrograms = np.array(train_mel_spectrograms)
train_mfccs = np.array(train_mfccs) 
train_y = to_categorical(np.array(train_y))

test_X = np.vstack(test_X)
test_spectrograms = np.array(test_spectrograms)
test_mel_spectrograms = np.array(test_mel_spectrograms)
test_mfccs = np.array(test_mfccs)
test_y = to_categorical(np.array(test_y))

print('train_X:', train_X.shape)
print('train_spectrograms:', train_spectrograms.shape)
print('train_mel_spectrograms:', train_mel_spectrograms.shape)
print('train_mfccs:', train_mfccs.shape)
print('train_y:', train_y.shape)
print()
print('test_X:', test_X.shape)
print('test_spectrograms:', test_spectrograms.shape)
print('test_mel_spectrograms:', test_mel_spectrograms.shape)
print('test_mfccs:', test_mfccs.shape)
print('test_y:', test_y.shape)

ip = Input(shape=(train_X[0].shape))
hidden = Dense(128, activation='relu')(ip)
op = Dense(10, activation='softmax')(hidden)
model = Model(input=ip, output=op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_X,
          train_y,
          epochs=10,
          batch_size=32,
          validation_data=(test_X, test_y))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

train_X_ex = np.expand_dims(train_spectrograms, -1)
test_X_ex = np.expand_dims(test_spectrograms, -1)
print('train X shape:', train_X_ex.shape)
print('test X shape:', test_X_ex.shape)

ip = Input(shape=train_X_ex[0].shape)
m = Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same')(ip)
m = MaxPooling2D(pool_size=(4, 4))(m)
m = Dropout(0.2)(m)
m = Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4, 4))(m)
m = Dropout(0.2)(m)
m = Flatten()(m)
m = Dense(32, activation='relu')(m)
op = Dense(3, activation='softmax')(m)

model = Model(input=ip, output=op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history2 = model.fit(train_X_ex,
          train_y,
          epochs=10,
          batch_size=32,
          verbose=1,
          validation_data=(test_X_ex, test_y))

plt.plot(history2.history['accuracy'], label='Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

train_X_ex2 = np.expand_dims(train_mel_spectrograms, -1)
test_X_ex2 = np.expand_dims(test_mel_spectrograms, -1)
print('train X shape:', train_X_ex2.shape)
print('test X shape:', test_X_ex2.shape)

ip = Input(shape=train_X_ex2[0].shape)
m = Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same')(ip)
m = MaxPooling2D(pool_size=(4, 4))(m)
m = Dropout(0.2)(m)
m = Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4, 4))(m)
m = Dropout(0.2)(m)
m = Flatten()(m)
m = Dense(32, activation='relu')(m)
op = Dense(3, activation='softmax')(m)

model = Model(input=ip, output=op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history3 = model.fit(train_X_ex2,
          train_y,
          epochs=10,
          batch_size=32,
          verbose=1,
          validation_data=(test_X_ex2, test_y))

plt.plot(history3.history['accuracy'], label='Train Accuracy')
plt.plot(history3.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

train_X_ex3 = np.expand_dims(train_mfccs, -1)
test_X_ex3 = np.expand_dims(test_mfccs, -1)
print('train X shape:', train_X_ex3.shape)
print('test X shape:', test_X_ex3.shape)

ip = Input(shape=train_X_ex3[0].shape)
m = Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
m = MaxPooling2D(pool_size=(4, 4))(m)
m = Flatten()(m)
m = Dense(32, activation='relu')(m)
op = Dense(3, activation='softmax')(m)

model = Model(input=ip, output=op)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history4 = model.fit(train_X_ex3,
          train_y,
          epochs=100,
          batch_size=32,
          verbose=0,
          validation_data=(test_X_ex3, test_y))
 
plt.plot(history4.history['accuracy'], label='Train Accuracy')
plt.plot(history4.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

corrects, wrongs = model.evaluate(train_X_ex3, train_y)
print("accuracy train: ", corrects / ( corrects + wrongs))
corrects, wrongs = model.evaluate(test_X_ex3, test_y)
print("accuracy: test", corrects / ( corrects + wrongs))

cm = model.confusion_matrix(train_X_ex3, train_y)
