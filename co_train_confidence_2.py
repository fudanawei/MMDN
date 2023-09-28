# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:54:28 2022
@author: Wei Zhou, Ziwei Li
DMNN: Dynamic learning multi-memory neural network for image recovery through unstablized MMF
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datafolder", type=str, default='../Data/gray3/20230826145147-rawdata-fr100-randPattern32_bin12_100000_gray3_V1_a16_wbg_round-exp100')
parser.add_argument("--saverootfolder", type=str, default="../Data/gray3/20230826145147-rawdata-fr100-randPattern32_bin12_100000_gray3_V1_a16_wbg_round-exp100/result")
parser.add_argument("--configFile", type=str, default="./config.yaml")
parser.add_argument("--use_model", type=str, default="pretrain", help="pretrain model prefix")
parser.add_argument("--modelfolder", type=str, default=None)
parser.add_argument("--graylevel", type=int, default=2)
parser.add_argument("--pretrain_state", type=int, default=1, help="state: 1 for pretrain, 0 for update")
parser.add_argument("--update_step", type=int, default=0, help="step number for update")
parser.add_argument("--sizeOfPretrain", type=int, default=20000, help="number of pre-training dataset")
parser.add_argument("--sizeOfBatch", type=int, default=1000, help="number of dataset for each update")
parser.add_argument("--sizeOfUpdateInvertal", type=int, default=1000, help="update interval, 10s * 100fps")
parser.add_argument("--flag_dynamic_learning", type=int, default=1)
parser.add_argument("--flag_multi_model", type=int, default=1)
parser.add_argument("--flag_rebuild", type=int, default=1)
parser.add_argument("--rebuild_interval", type=int, default=10)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--speckle_dim", type=int, default=150)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# folder names
datafolder = args.datafolder
saverootfolder = args.saverootfolder
configFile = args.configFile
modelfolder = args.modelfolder
use_model = args.use_model

# training set
pretrain_state = args.pretrain_state   # state: 1 for pretrain, 0 for update
update_step = args.update_step   # step number for update, default: 0
sizeOfPretrain = args.sizeOfPretrain  # number of pre-training dataset
sizeOfBatch = args.sizeOfBatch        # number of dataset for each update, default=1000 
sizeOfUpdateInvertal = args.sizeOfUpdateInvertal  # update interval, 10s * 100fps
initialDataLength = update_step * sizeOfUpdateInvertal
finalDataLength = initialDataLength + sizeOfPretrain

# mode
flag_dynamic_learning = args.flag_dynamic_learning
flag_multi_model = args.flag_multi_model
flag_rebuild = args.flag_rebuild and flag_multi_model and flag_dynamic_learning
rebuild_interval = args.rebuild_interval

# input image & output image
speckle_dim = args.speckle_dim
graylevel = args.graylevel

## --------------------------------------------------------------- ##
#  
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, save_model
import matplotlib.pyplot as plt
from model import createCNN1, get_callbacks
from utils import write_accuracies_to_xls, save_diffimages, save_average_diffimages
from model_cotrain import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
# disable tensorflow debugging info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

with open(configFile, 'r', encoding='utf-8') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    # set optimizer
    ada = keras.optimizers.Adadelta(lr=config['lr_rate'])
    epochs = config['epochs']
    batch_size = config['batch_size']

## --------------------------------------------------------------- ##
#                          data loading
## --------------------------------------------------------------- ##

# load target pattern
dataLabels = np.load(os.path.join(datafolder, 'pattern.npy')).astype(np.uint8)    # P x N2
outsize = dataLabels.shape[1]

# load input speckle
dataValues0 = np.load(os.path.join(datafolder, 'speckles.npy')).astype(np.uint8)  # P x M2
# resize speckle images
dataValues = []
for i in range(dataValues0.shape[0]):
    dataValues.append(np.resize(dataValues0[i,:,:], (speckle_dim,speckle_dim)))
dataValues = np.array(dataValues)
input_shape = (speckle_dim, speckle_dim, 1)
dataValues = dataValues.reshape(dataValues.shape[0], dataValues.shape[1], dataValues.shape[2], 1)

print('[data set size: ', dataValues.shape, ' -> ', dataLabels.shape, ']')

# description of update
batches = int((dataValues.shape[0] - finalDataLength) / sizeOfUpdateInvertal) 
if pretrain_state:
    print(f'[Pre-train: dataset number={sizeOfPretrain:d}]')
print(f'[Update: dataset number={sizeOfBatch:d}, update interval={sizeOfUpdateInvertal:d}, update steps={batches:d}]')

# save path
if not os.path.exists(saverootfolder):
    os.mkdir(saverootfolder)
print('Save folder: ', saverootfolder)          

resultfolder = os.path.join(saverootfolder, 'result')
if not os.path.exists(resultfolder):
    os.mkdir(resultfolder)

writepath_acc = os.path.join(saverootfolder, 'co_train_accuracy_excel')
if not os.path.exists(writepath_acc):
    os.mkdir(writepath_acc)

writepath_diffimage = os.path.join(saverootfolder, 'co_train_accuracy_image')
if not os.path.exists(writepath_diffimage):
    os.mkdir(writepath_diffimage)

# to record grayscale prediction and spatial difference
diffimage,diffimage_M1,diffimage_M2,diffimage_M3=[],[],[],[]
pre_M1,pre_M2,pre_M3 = [],[],[]


## --------------------------------------------------------------- ##
#                         model initializing
## --------------------------------------------------------------- ##
# init models
clf1 = createCNN1(outsize=outsize, input_shape=input_shape)
clf1.compile(loss=keras.losses.binary_crossentropy, optimizer=ada)
clf2 = createCNN1(outsize=outsize, input_shape=input_shape)
clf2.compile(loss=keras.losses.binary_crossentropy, optimizer=ada)
clf3 = createCNN1(outsize=outsize, input_shape=input_shape)
clf3.compile(loss=keras.losses.binary_crossentropy, optimizer=ada)

# load pre-train model
if use_model is not None:
    print('Load model from: ', os.path.join(modelfolder, use_model))
    try:
        clf3 = load_model(os.path.join(modelfolder, use_model+'_clf3.hdf5'))
    except:
        print('Error: model not exist!')
        use_model = None
    try:
        if flag_multi_model:
            clf2 = load_model(os.path.join(modelfolder, use_model+'_clf2.hdf5'))
            clf1 = load_model(os.path.join(modelfolder, use_model+'_clf1.hdf5'))
        else:
            clf2 = clf3
            clf1 = clf3
    except:
        print('Error: model not exist!')


### --------------------------------------------------------###
### ------------------- dynamic learning -------------------###
### --------------------------------------------------------###

# labeled trainning data
x_labeled = np.copy(dataValues[initialDataLength:finalDataLength])
if pretrain_state or not flag_dynamic_learning:
    y_labeled = np.copy(dataLabels[initialDataLength:finalDataLength])
else:
    print('Resume, load predicted label data')
    import glob
    pred_y_files = glob.glob(os.path.join(resultfolder, 'predictY_*.npy'))
    pred_y_files.sort()
    y_labeled = dataLabels[:20000]
    for f in pred_y_files:
        dataLabels_pred = np.load(f)  # should use predicted y instead of gt y
        y_labeled = np.concatenate((y_labeled, dataLabels_pred), axis=0)
    y_labeled = np.copy(y_labeled[initialDataLength:finalDataLength])
    print("y_labeled: ", y_labeled.shape)


for t in range(update_step+1, batches+update_step+1):
    print('='*20, 'step:', t,'/',batches+update_step,'='*20)
    if pretrain_state:
        print(f'[Train set: {finalDataLength-sizeOfPretrain} - {finalDataLength}]')
    else:
        print(f'[Train set: {finalDataLength-sizeOfBatch} - {finalDataLength}]')
    
    # testing / validation data
    initialDataLength = finalDataLength
    finalDataLength = finalDataLength + sizeOfUpdateInvertal  # move one step forward
    x_fortest = np.copy(dataValues[initialDataLength:finalDataLength])
    y_fortest = np.copy(dataLabels[initialDataLength:finalDataLength])
    print(f'[Test set : {initialDataLength} - {finalDataLength}]')
    
    # remove reference pattern inserted at 100-frame interval
    x_fortest[::100] = 0
    y_fortest[::100] = 0
    x_labeled[::100] = 0
    y_labeled[::100] = 0

    # rebuild short-term model1: compare w and w/o starting from scratch
    if flag_rebuild:
        if t % (2*rebuild_interval) == rebuild_interval+1:
            print("="*10 + "Rebuild Model_1" + "="*10)
            # load pre-train model
            if use_model is not None and 0:
                clf1 = load_model(os.path.join(modelfolder, 'pretrain_clf1.hdf5'))
            else:
                clf1 = createCNN1(outsize=outsize, input_shape=input_shape)
                clf1.compile(loss=keras.losses.binary_crossentropy, optimizer=ada)
            clf1.fit(x_labeled[-int(sizeOfPretrain/3*2):], y_labeled[-int(sizeOfPretrain/3*2):], batch_size=batch_size, epochs=int(2*epochs), callbacks=get_callbacks(), validation_data=(x_fortest, y_fortest))        
        # rebuild short-term model2
        if t % (2*rebuild_interval) == 1 and t > 1:
            print("="*10 + "Rebuild Model_2" + "="*10)
            # load pre-train model
            if use_model is not None and 0:
                clf2 = load_model(os.path.join(modelfolder, 'pretrain_clf2.hdf5'))
            else:
                clf2 = createCNN1(outsize=outsize, input_shape=input_shape)
                clf2.compile(loss=keras.losses.binary_crossentropy, optimizer=ada)
            clf2.fit(x_labeled[-int(sizeOfPretrain/4*3):], y_labeled[-int(sizeOfPretrain/4*3):], batch_size=batch_size, epochs=int(2*epochs), callbacks=get_callbacks(), validation_data=(x_fortest, y_fortest))

    # pre-training: 如果多次测试且每次都覆盖pretrain.h5，太好的pretrain model反而后面的表现变差？
    if pretrain_state and flag_dynamic_learning:
        print("="*10 + " Network pre-training " + "="*10)
        print("-"*4 + "> model 3")
        clf3.fit(x_labeled[-sizeOfPretrain:], y_labeled[-sizeOfPretrain:], batch_size=batch_size, epochs=int(epochs*2),callbacks=get_callbacks(), validation_data=(x_fortest, y_fortest))
        if not os.path.exists(os.path.join(modelfolder, 'pretrain_clf3.hdf5')):  # TODO: decide whether we save the pretrain model
            save_model(clf3, os.path.join(modelfolder, 'pretrain_clf3.hdf5'))
        else:
            print("Ssve updated pretrain model as pretrain_round2.h5")
            save_model(clf3, os.path.join(modelfolder, 'pretrain_round2_clf3.hdf5'))
        if flag_multi_model:
            print("-"*4 + "> model 2")  # clf1, clf2 are different from clf3 by training on smaller datasets
            clf2.fit(x_labeled[-int(sizeOfPretrain/4*3):], y_labeled[-int(sizeOfPretrain/4*3):], batch_size=batch_size, epochs=int(epochs*2),callbacks=get_callbacks(),validation_data=(x_fortest, y_fortest))
            if not os.path.exists(os.path.join(modelfolder, 'pretrain_clf2.hdf5')):
                save_model(clf2, os.path.join(modelfolder, 'pretrain_clf2.hdf5'))
            else:
                save_model(clf2, os.path.join(modelfolder, 'pretrain_round2_clf2.hdf5'))            
            print("-"*4 + "> model 1")
            clf1.fit(x_labeled[-int(sizeOfPretrain*2/3):], y_labeled[-int(sizeOfPretrain*2/3):], batch_size=batch_size, epochs=int(epochs*2),callbacks=get_callbacks(),validation_data=(x_fortest, y_fortest))
            if not os.path.exists(os.path.join(modelfolder, 'pretrain_clf1.hdf5')):
                save_model(clf1, os.path.join(modelfolder, 'pretrain_clf1.hdf5'))
            else:
                save_model(clf1, os.path.join(modelfolder, 'pretrain_round2_clf1.hdf5'))
        else:
            clf2 = clf3
            clf1 = clf3
    # fine-tuning
    elif flag_dynamic_learning:
        print("="*10 + " Network update " + "="*10)
        print("-"*4 + "> model 3")
        clf3.fit(x_labeled[-sizeOfBatch:], y_labeled[-sizeOfBatch:], batch_size=batch_size, epochs=int(epochs), callbacks=get_callbacks(),validation_data=(x_fortest, y_fortest))
        if flag_multi_model:
            print("-"*4 + "> model 2")
            clf2.fit(x_labeled[-sizeOfBatch:], y_labeled[-sizeOfBatch:], batch_size=batch_size, epochs=int(epochs), callbacks=get_callbacks(), validation_data=(x_fortest, y_fortest))
        else: 
            clf2 = clf3
        if flag_multi_model:
            print("-"*4 + "> model 1")
            clf1.fit(x_labeled[-sizeOfBatch:], y_labeled[-sizeOfBatch:], batch_size=batch_size, epochs=int(epochs), callbacks=get_callbacks(), validation_data=(x_fortest, y_fortest))
        else:
            clf1 = clf3

    # model co-inference and evaluation
    predsb,accuracy_M1,accuracy_M2,accuracy_M3,accuracy_Ensem,conf1,conf2 = co_train_evluation(clf1, clf2, clf3, x_fortest, y_fortest, pre_M1,pre_M2,pre_M3)
    
    # update training data for the next round update
    x_labeled = np.concatenate((x_labeled, x_fortest), axis=0)[-sizeOfPretrain:]  # keep the last sizeOfPretrain (in case of rebuild)
    y_labeled = np.concatenate((y_labeled, predsb), axis=0) # always append

    # save predicted y at current timestamp
    np.save(os.path.join(resultfolder, f'predictY_{initialDataLength:06d}'), predsb.astype(np.uint8))
    
    pretrain_state = 0 # switch to update mode
    
    # write accuracy to accuracy.xls file
    write_accuracies_to_xls(writepath_acc, t, accuracy_M1, accuracy_M2, accuracy_M3, accuracy_Ensem, conf1, conf2)
    print("T={}: ensemble acc is {:.3f}, self acc is {:.3f}, {:.3f}, {:.3f}".format(t, accuracy_Ensem[1],accuracy_M1[1],accuracy_M2[1],accuracy_M3[1]))

    # write absolute difference of recovered image
    # diffimage.append(accuracy_Ensem[2])
    # diffimage_M1.append(accuracy_M1[2])
    # diffimage_M2.append(accuracy_M2[2])
    # diffimage_M3.append(accuracy_M3[2])
    # save_diffimages(writepath_diffimage, t, accuracy_Ensem[2],accuracy_M1[2],accuracy_M2[2],accuracy_M3[2])

    # cache intermediate results in case of program crash 
    if flag_multi_model and flag_dynamic_learning and (t%(2*rebuild_interval)==0):
        print("<=== cache intermediate model and prediction ===>")
        try:
            print("model 1")
            clf1.save(os.path.join(modelfolder, 'update_' + str(t) + '_clf1.hdf5'))
            print("model 2")
            clf2.save(os.path.join(modelfolder, 'update_' + str(t) + '_clf2.hdf5'))
            print("model 3")
            clf3.save(os.path.join(modelfolder, 'update_' + str(t) + '_clf3.hdf5'))
        except Exception as e:
            print("Error occurs when saving model!")
            print(repr(e))

        # np.save(os.path.join(resultfolder, f'y_labeled_0_{finalDataLength}.npy'), np.array(y_labeled))
        if flag_multi_model:
            np.save(os.path.join(resultfolder, 'pre_M1.npy'), np.array(pre_M1))
            np.save(os.path.join(resultfolder, 'pre_M2.npy'), np.array(pre_M2))
            np.save(os.path.join(resultfolder, 'pre_M3.npy'), np.array(pre_M3))    

np.save(os.path.join(resultfolder, f'y_enblem_step{update_step:d}-{batches+update_step:d}.npy'), np.array(y_labeled))

# average difference of diffimage
# save_average_diffimages(writepath_diffimage,diffimage,diffimage_M1,diffimage_M2,diffimage_M3)
