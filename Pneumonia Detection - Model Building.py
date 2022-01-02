# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:56:13 2021

@author: Himanshu
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
##Import any other stats/DL/ML packages you may need here. E.g. Keras, scikit-learn, etc.
from itertools import chain
from random import sample 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, plot_precision_recall_curve, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import binarize
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50 
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


## Below is some helper code to read all of your full image filepaths into a dataframe for easier manipulation
## Load the NIH data to all_xray_df
all_xray_df = pd.read_csv('F:/Chest Xrays/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('F:/Chest Xrays/','images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

## Create some extra columns in your table with binary indicators of certain diseases 
## rather than working directly with the 'Finding Labels' column

df = all_xray_df.copy()

all_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))

for c_label in all_labels:
    class_label = c_label + "_class"
    if len(c_label)>1: # leave out empty labels
        df[class_label.lower()] = df['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)

def create_splits(df, test_size, target_column):
    
    ## Either build your own or use a built-in library to split your original dataframe into two sets 
    ## that can be used for training and testing your model
    ## It's important to consider here how balanced or imbalanced you want each of those sets to be
    ## for the presence of pneumonia
    
    # Todo
    
    train_data, val_data = train_test_split(df, 
                                   test_size = test_size, 
                                   stratify = df[target_column])
    
    return train_data, val_data

train_df, val_df = create_splits(df, 0.2, 'pneumonia_class')
print('train_df.shape:', train_df.shape)
print('val_df.shape:', val_df.shape)

# Balancing training set
p_inds = train_df[train_df.pneumonia_class==1].index.tolist()
np_inds = train_df[train_df.pneumonia_class==0].index.tolist()

np_sample = sample(np_inds,len(p_inds))
train_df = train_df.loc[p_inds + np_sample]


# Balancing validation set
p_inds = val_df[val_df.pneumonia_class==1].index.tolist()
np_inds = val_df[val_df.pneumonia_class==0].index.tolist()

np_sample = sample(np_inds,4*len(p_inds))
val_df = val_df.loc[p_inds + np_sample]


def my_image_augmentation():
    
    ## recommendation here to implement a package like Keras' ImageDataGenerator
    ## with some of the built-in augmentations 
    
    ## keep an eye out for types of augmentation that are or are not appropriate for medical imaging data
    ## Also keep in mind what sort of augmentation is or is not appropriate for testing vs validation data
    
    # Todo
    my_idg = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.1, 
                              width_shift_range=0.1, 
                              rotation_range=20, 
                              shear_range = 0.1,
                              zoom_range=0.1)
    
    return my_idg


def make_train_gen(target_size, batch_size):
    
    ## Create the actual generators using the output of my_image_augmentation for your training data
    ## Suggestion here to use the flow_from_dataframe library, e.g.:
    my_train_idg = my_image_augmentation()
    train_gen = my_train_idg.flow_from_dataframe(dataframe=train_df, 
                                         directory=None, 
                                         x_col = "path",
                                         y_col = 'pneumonia_class',
                                         class_mode = 'raw',
                                         target_size = target_size, 
                                         batch_size = batch_size
                                         )

    return train_gen


def make_val_gen(target_size, batch_size):
    
    my_val_idg = ImageDataGenerator(rescale=1. / 255.0)
    val_gen = my_val_idg.flow_from_dataframe(dataframe=val_df, 
                                             directory=None, 
                                             shuffle = False,
                                             x_col = "path",
                                             y_col = 'pneumonia_class',
                                             class_mode = 'raw',
                                             target_size = target_size, 
                                             batch_size = batch_size
                                             )
    
    return val_gen

batch_size = 64
target_size = (224, 224)

train_gen = make_train_gen(target_size, batch_size)
val_gen = make_val_gen(target_size, batch_size)

## May want to pull a single large batch of random validation data for testing after each epoch:
valX, valY = val_gen.next()


# Build ing Model

from tensorflow.keras.layers import Input, Conv2D 
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense 
from tensorflow.keras import Model

def load_pretrained_model():
    
#     model = ResNet50(include_top=False, weights='imagenet')
# #     transfer_layer = model.get_layer('conv5_block3_out')
#     new_model = Model(inputs = model.input, outputs = model.output)
    
#     for layer in new_model.layers[:-10]:
#         layer.trainable = False
    
    model = VGG16(include_top=True, weights='imagenet')
    
    transfer_layer = model.get_layer('block5_pool')
    vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)
    
    for layer in vgg_model.layers[:17]:
        layer.trainable = False
        
    return vgg_model
#     return new_model


def build_model():
    # input
    # Fully connected layers
    my_model = load_pretrained_model()
    pd_model = Sequential([
        my_model,
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='relu')
    ])
    
    return pd_model
my_model = build_model()
print(my_model.summary())

## Below helper code will allow you to add checkpoints to your model,
## This will save the 'best' version of your model by comparing it to previous epochs of training

## Note that you need to choose which metric to monitor for your model's 'best' performance if using this code. 
## The 'patience' parameter is set to 10, meaning that your model will train for ten epochs without seeing
## improvement before quitting

weight_path="{}_my_model.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, 
                             monitor= 'val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode= 'min', 
                             save_weights_only = True)

early = EarlyStopping(monitor= 'val_loss', 
                      mode= 'min', 
                      patience=10)

callbacks_list = [checkpoint, early]

## train your model

my_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
history = my_model.fit(train_gen, 
                          validation_data = (valX, valY), 
                          epochs = 20, 
                          callbacks = callbacks_list)
