# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 15:48
# @Author  : sunjiawei
# @FileName: trainDAN_Menpo.py


##训练部分的代码
import os
os.environ["CUDA_VISIBLE_DEVICES"]='5'

import tensorflow as tf
import numpy as np

from ImageServer import ImageServer
from models_test2 import DAN
import cv2

datasetDir = "../data/result/"
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=192540_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=200_perturbations=[]_size=[112, 112].npz")

# Create a new DAN regressor.
# 两个stage训练，不宜使用 estimator
# regressor = tf.estimator.Estimator(model_fn=DAN, params={})

def evaluateError(landmarkGt, landmarkP):
    e = np.zeros(68)
    ocular_dist = np.mean(np.linalg.norm(landmarkGt[36:42] - landmarkGt[42:48], axis=1))
    for i in range(68):
        e[i] = np.linalg.norm(landmarkGt[i] - landmarkP[i])
    e = e / ocular_dist
    return e

## 计算每个batch的误差
def evaluateBatchError(landmarkGt, landmarkP, batch_size):
    e = np.zeros([batch_size, 68])
    for i in range(batch_size):
        e[i] = evaluateError(landmarkGt[i], landmarkP[i])
    mean_err = e[:,:].mean()#axis=0)
    return mean_err

## 每个数据样本的标签---其实是68个关键点的坐标位置
def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))

nSamples = trainSet.gtLandmarks.shape[0]
imageHeight = trainSet.imgSize[0]
imageWidth = trainSet.imgSize[1]
nChannels = trainSet.imgs.shape[3]

Xtrain = trainSet.imgs
Xvalid = validationSet.imgs
# import pdb; pdb.set_trace()

Ytrain = getLabelsForDataset(trainSet)
Yvalid = getLabelsForDataset(validationSet)
testIdxsTrainSet = range(len(Xvalid))
testIdxsValidSet = range(len(Xvalid))
meanImg = trainSet.meanImg
stdDevImg = trainSet.stdDevImg
initLandmarks = trainSet.initLandmarks[0].reshape((1, 136))

S1_isTrain = tf.placeholder(tf.bool)
S2_isTrain = tf.placeholder(tf.bool)

dan = DAN(initLandmarks)

STAGE = 1

with tf.Session() as sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", sess.graph)
    if STAGE < 2:
        sess.run(tf.global_variables_initializer())
        tf.train.write_graph(sess.graph_def, "./DAN_Menpo_Model/", "DAN_Menpo_graph.pb", as_text=False)
    else:
        Saver.restore(sess, './DAN_Menpo_Model/DAN-Menpo.ckpt')
        print('Pre-trained model has been loaded!')

    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    print("Starting training......")
    for epoch in range(150):
        Count = 0
        while Count * 128 < Xtrain.shape[0]:
            RandomIdx = np.random.choice(Xtrain.shape[0], 128, False)
            if STAGE == 1 or STAGE == 0:
                sess.run(dan['S1_Optimizer'],feed_dict={dan['InputImage']: Xtrain[RandomIdx], dan['GroundTruth']: Ytrain[RandomIdx], dan['S1_isTrain']: True,dan['S2_isTrain']: False})
            else:
                sess.run(dan['S2_Optimizer'],feed_dict={dan['InputImage']: Xtrain[RandomIdx], dan['GroundTruth']: Ytrain[RandomIdx],dan['S1_isTrain']: False, dan['S2_isTrain']: True})

            if Count % 128 == 0:
                TestErr = 0
                BatchErr = 0
                if STAGE == 1 or STAGE == 0:
                    TestErr = sess.run(dan['S1_Cost'],{dan['InputImage']: Xvalid, dan['GroundTruth']: Yvalid, dan['S1_isTrain']: False, dan['S2_isTrain']: False})
                    # print(evaluateBatchError(Yvalid.reshape([-1, 68, 2]), S1_Ret.reshape([-1, 68, 2]), 9))
                    BatchErr = sess.run(dan['S1_Cost'],{dan['InputImage']: Xtrain[RandomIdx], dan['GroundTruth']: Ytrain[RandomIdx],dan['S1_isTrain']: False, dan['S2_isTrain']: False})
                else:
                    # Landmark,Img,HeatMap,FeatureUpScale =sess.run([dan['S2_InputLandmark'],dan['S2_InputImage'],dan['S2_InputHeatmap'],dan['S2_FeatureUpScale']],{dan['InputImage']:Xtrain[RandomIdx],dan['GroundTruth']:Ytrain[RandomIdx],dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    TestErr = sess.run(dan['S2_Cost'],{dan['InputImage']: Xvalid, dan['GroundTruth']: Yvalid, dan['S1_isTrain']: False,dan['S2_isTrain']: False})
                    BatchErr = sess.run(dan['S2_Cost'],{dan['InputImage']: Xtrain[RandomIdx], dan['GroundTruth']: Ytrain[RandomIdx],dan['S1_isTrain']: False, dan['S2_isTrain']: False})
                print('Epoch: ', epoch, ' Batch: ', Count, 'TestErr:', TestErr, ' BatchErr:', BatchErr)
            Count += 1
        Saver.save(sess, './DAN_Menpo_Model/DAN-Menpo.ckpt')